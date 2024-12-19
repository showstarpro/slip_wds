# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
from collections import OrderedDict
from typing import Optional
import numpy as np
import timm
import torch
from torch import nn
import math
from torch.utils.checkpoint import checkpoint
from transformers import CLIPConfig, CLIPModel
from dataclasses import dataclass
import losses
from auxilary import MultiheadAttention
from diffusion import create_diffusion, gaussian_diffusion
from cls_model.transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
    MixClsHead,
)
from cls_model.model import CLIPTextCfg

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # self.attn = MultiheadAttention(d_model, n_head)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    #     self.attn_probs = None
    #     self.attn_grad = None

    # def set_attn_probs(self, attn_probs):
    #     self.attn_probs = attn_probs

    # def set_attn_grad(self, attn_grad):
    #     self.attn_grad = attn_grad


    # def attention(self, x: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    #     return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, attention_probs_forward_hook=self.set_attn_probs,
    #                     attention_probs_backwards_hook=self.set_attn_grad)[0]

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        x = self.visual(image)
        # x = x.pooler_output
        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}


class SIMCLR(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # ssl
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model

        self.image_mlp = self._build_mlp(in_dim=vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def encode_image(self, image):
        x = self.visual(image)

        return x

    def forward(self, aug1, aug2):
        h1 = self.visual(aug1)
        h2 = self.visual(aug2)

        aug1_embed = self.image_mlp(h1)
        aug2_embed = self.image_mlp(h2)

        return {'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed}


class SLIP(CLIP):
    def __init__(self,
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.image_mlp = self._build_mlp(in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def forward(self, image, text, aug1, aug2):
        aug1_embed = self.image_mlp(self.visual(aug1).pooler_output )
        aug2_embed = self.image_mlp(self.visual(aug2).pooler_output )
        
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp(),
                'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed}

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        # z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        # self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)

        y = t

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

@dataclass
class ClassHeadCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    layers: int = 1

def _build_cls_head(
        width,
        clshead_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    clshead_cfg = ClassHeadCfg(**clshead_cfg) if isinstance(clshead_cfg, dict) else clshead_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    head = MixClsHead(
        width=width,
        layers=clshead_cfg.layers,
        mlp_ratio=clshead_cfg.mlp_ratio,
        act_layer=act_layer,
        norm_layer=norm_layer,
        output_dim=clshead_cfg.vocab_size,
    )

    return head

class MultiTask(CLIP):
    def __init__(self,
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 text_cfg: CLIPTextCfg,
                 grad_checkpointing=False,
                 diffloss_d=3,
                 num_sampling_steps='100',
                 decoder_embed_dim=512,
                 quick_gelu: bool = False,
                 cast_dtype: Optional[torch.dtype] = None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.image_mlp = self._build_mlp(in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.token_embed_dim = 768
        self.decoder = SimpleMLPAdaLN(
            in_channels=self.token_embed_dim,
            model_channels=self.vision_width,
            out_channels=self.vision_width,
            # z_channels=decoder_embed_dim,
            num_res_blocks=diffloss_d,
            grad_checkpointing=grad_checkpointing
        )
        self.norm = nn.LayerNorm(self.vision_width)
        self.diffusion_batch_mul = 4
        clshead_cfg = ClassHeadCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        self.text_decoder = _build_cls_head(
            width=decoder_embed_dim,
            clshead_cfg=clshead_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )
        self.register_buffer("cap_fq", torch.zeros([1, self.vocab_size], dtype=torch.float64))
        self.register_buffer("num_samples", torch.zeros([1, 1], dtype=torch.float64))

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = 16
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x, C, H, W):
        bsz = x.shape[0]
        p = 16
        h, w = H // p, W // p

        x = x.reshape(bsz, h, w, C, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, C, h * p, w * p)
        return x  # [n, c, h, w]
    
    # def encode(self, image):
    #     x = self.visual(image)
    #     x = x.last_hidden_state
    #     x = self.norm(x)
    #     return x

    def forward(self, image, text):
        B, C, H, W = image.shape
        ## add noise
        img = self.patchify(image)
        b, l, d = img.shape
        img = img.reshape(b * l, -1)
        noise = torch.randn_like(img) # [b*l, d] 
        ## 不同等级系数
        # alpha = torch.rand(b, l).to(device=noise.device)
        # noise = noise * alpha.unsqueeze(-1)
        # noise = noise.reshape(b, l, -1)
        # noise = self.unpatchify(noise, C, H, W)
        t = torch.randint(0, self.train_diffusion.num_timesteps, (img.shape[0],), device=img.device) # [b*l]
        x_t = self.train_diffusion.q_sample(img, t, noise) # [b*l, d]
        x_t = x_t.reshape(b, l, -1)
        x_t = self.unpatchify(x_t, C, H, W) # recover shape [B, C, H, W]

        img1 = self.visual(x_t) ## with noise
        img2 = self.visual(image) ## without noise

        ## diffusion
        # imgnoise_embed = self.norm(img1.last_hidden_state) ## feature with noise
        imgnoise_embed = self.norm(self.visual.forward_features(x_t))
        # image_embed = self.encode(image) ## without noise
        ## simclr
        # aug1_embed = self.image_mlp(img1.pooler_output)
        # aug2_embed = self.image_mlp(img2.pooler_output)
        aug1_embed = self.image_mlp(img1)
        aug2_embed = self.image_mlp(img2)
        ## clip and superclass
        img_embed_p = self.encode_image(image)
        # imgnoise_embed_p = self.encode_image(x_t)
        text_embed = self.encode_text(text)

        ## add mar mlp as decoder, reshape操作参考mar code
        B, L, _ = imgnoise_embed[:, 1:].shape
        n_imgnoise_embed = imgnoise_embed[:, 1:].reshape(B * L, -1)
        # n_image_embed = image_embed[:, 1:].reshape(B * L, -1)
        # n_t = torch.randint(0, self.train_diffusion.num_timesteps, (n_image_embed.shape[0],), device=n_image_embed.device)
        # model_kwargs = dict(c=None) ## none
        # alpha = alpha.reshape(B * L, -1).squeeze(-1)
        rec_noise = self.decoder(n_imgnoise_embed, t) ## recover noise
        # noise = self.patchify(noise)
        # B, L, _ = noise.shape
        # noise = noise.reshape(B * L, -1)

        ## add img to text decode
        text_tokens = self.text_decoder(img_embed_p)
        labels = text.clone()

        return {'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed,
                'img_embed_p': img_embed_p,
                'text_embed': text_embed,
                'noise': noise,
                'rec_noise': rec_noise,
                'text_tokens': text_tokens,
                'labels': labels,
                "cap_fq": self.cap_fq,
                "num_samples": self.num_samples,
                'logit_scale': self.logit_scale.exp()}

class LinearModel(nn.Module):
    def __init__(
        self,
        model,
        num_classes=1000,
    ):
        super().__init__()
        self.visual = model
        self.num_classes = num_classes
        self.vision_width = model.config.hidden_size
        self.norm = nn.LayerNorm(self.vision_width)
        self.head_drop = nn.Dropout(0.)
        self.head = nn.Linear(self.vision_width, self.num_classes)

    def forward(self, image):
        x = self.visual(image).pooler_output
        x = self.norm(x)
        x = self.head_drop(x)
        output = self.head(x)

        return output

# Define a new CLIP configuration
config = CLIPConfig(
    text_config=dict(
        vocab_size=50265,  # Common vocabulary size for text
        hidden_size=512,   # Size of hidden layers
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        layer_norm_eps=1e-5
    ),
    vision_config=dict(
        hidden_size=768,   # Size of hidden layers for vision input
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=224,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-5
    ),
    projection_dim=512      # Projection size for similarity calculation
)

def get_loss(model, ssl_temp, clip_scale, ssl_scale, diff_scale, cls_scale):
    if model.startswith('MultiTask'):
        ssl_loss = losses.SIMCLRLoss(temperature=ssl_temp)
        return losses.MultiTaskLoss(ssl_loss, clip_scale, ssl_scale, diff_scale, cls_scale)
    if model.startswith('SLIP'):
        ssl_loss = losses.SIMCLRLoss(temperature=ssl_temp)
        return losses.SLIPLoss(ssl_loss, ssl_scale)
    if model.startswith('CLIP'):
        return losses.CLIPLoss()
    if model.startswith('SIMCLR'):
        return losses.SIMCLRLoss(temperature=ssl_temp)


def get_metric_names(model):
    if model.startswith('SLIP'):
        return ['loss', 'clip_loss', 'ssl_loss', 'clip_acc', 'ssl_acc']
    elif model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    elif model.startswith('MultiTask'):
        return ['loss', 'clip_loss', 'ssl_loss', 'diff_loss', 'cls_loss', 'clip_acc', 'ssl_acc']
    else:
        return ['loss', 'ssl_loss', 'ssl_acc']


@timm.models.register_model
def vit_small_mocov3_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=12, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer('vit_small_patch16_224', **model_kwargs)

    return model


def CLIP_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = CLIP(embed_dim=512, vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=384, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def CLIP_VITB16(**kwargs):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model_clip_transformer = CLIPModel(config)
    vision_model = model_clip_transformer.vision_model
    model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model

def MultiTask_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    # model_clip_transformer = CLIPModel(config)
    # vision_model = model_clip_transformer.vision_model
    model = MultiTask(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, text_cfg={}, **kwargs)

    return model

def LinearModel_VITB16(**kwargs):
    model_clip_transformer = CLIPModel(config)
    vision_model = model_clip_transformer.vision_model
    model = LinearModel(model=vision_model, **kwargs)

    return model

def SIMCLR_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=768, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITB16(**kwargs):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model_clip_transformer = CLIPModel(config)
    vision_model = model_clip_transformer.vision_model
    model = SLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def CLIP_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = CLIP(embed_dim=512, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=1024, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model
