# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

import utils


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['img_embed_p']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_loss': loss, 'clip_acc': acc}


class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        q_a = outputs['aug1_embed']
        q_b = outputs['aug2_embed']

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)

        k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}


class SLIPLoss(nn.Module):
    def __init__(self, ssl_loss, ssl_scale):
        super().__init__()
        self.clip_loss = CLIPLoss()
        self.ssl_loss = ssl_loss
        self.ssl_scale = ssl_scale

    def forward(self, outputs):
        clip_loss_dict = self.clip_loss(outputs)
        clip_loss = clip_loss_dict['clip_loss']
        clip_acc = clip_loss_dict['clip_acc']

        ssl_loss_dict = self.ssl_loss(outputs)
        ssl_loss = ssl_loss_dict['ssl_loss']
        ssl_acc = ssl_loss_dict['ssl_acc']

        return {'loss': clip_loss + self.ssl_scale * ssl_loss,
                'clip_loss': clip_loss,
                'clip_acc': clip_acc,
                'ssl_loss': ssl_loss,
                'ssl_acc': ssl_acc}

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class DiffLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        rec_noise = outputs['rec_noise']
        noise = outputs['noise']
        ## L2 loss
        loss = mean_flat((noise - rec_noise) ** 2).mean()

        return {'loss':loss, 'diff_loss':loss}

class CLSLoss(nn.Module):
    def __init__(
            self,
            world_size,
            pad_id=0,  # pad_token for open_clip custom tokenizer
        ):
        super().__init__()
        self.pad_id = pad_id
        self.world_size = world_size

    def loss(self, logits, targets):
        norm_item = F.normalize(targets, p=1, dim=1)
        loss =  -(F.log_softmax(logits, dim=1) * norm_item).sum(dim=1).mean()
        return loss
    
    def reweight_targets(self, cap_fq, num_samples, targets):
        cap_fq += targets.sum(dim=0, keepdim=True) / targets.shape[0]
        num_samples += 1
        dist.all_reduce(cap_fq, op=dist.ReduceOp.AVG)
        dist.all_reduce(num_samples, op=dist.ReduceOp.AVG)
        all_batch_size = self.world_size * targets.shape[0]
        targets = targets * torch.log((num_samples+1.0/all_batch_size) / (cap_fq+1.0/all_batch_size)).to(dtype=targets.dtype)
        return targets

    def forward(self, outputs):
        labels = outputs['labels']
        text_tokens = outputs['text_tokens']
        cap_fq = outputs['cap_fq']
        num_samples = outputs['num_samples']
        B, C = text_tokens.shape
        targets = torch.zeros(B, C, dtype=torch.float32).to(labels.device)
        # scatter labels to one-hot
        targets.scatter_(dim=1, index=labels, value=1.0)
        targets = self.reweight_targets(cap_fq, num_samples, targets)
        loss = self.loss(text_tokens, targets)

        return {'loss': loss, 'cls_loss': loss}

class MultiTaskLoss(nn.Module):
    def __init__(self, ssl_loss, ssl_scale, diff_scale, cls_scale):
        super().__init__()
        self.clip_loss = CLIPLoss()
        self.ssl_loss = ssl_loss
        self.ssl_scale = ssl_scale
        self.diff_loss = DiffLoss()
        self.diff_scale = diff_scale
        self.cls_loss = CLSLoss(world_size=1)
        self.cls_scale = cls_scale

    def forward(self, outputs):
        clip_loss_dict = self.clip_loss(outputs)
        clip_loss = clip_loss_dict['clip_loss']
        clip_acc = clip_loss_dict['clip_acc']

        ssl_loss_dict = self.ssl_loss(outputs)
        ssl_loss = ssl_loss_dict['ssl_loss']
        ssl_acc = ssl_loss_dict['ssl_acc']

        diff_loss_dict = self.diff_loss(outputs)
        diff_loss = diff_loss_dict['diff_loss']

        cls_loss_dict = self.cls_loss(outputs)
        cls_loss = cls_loss_dict['cls_loss']

        return {'loss': clip_loss + self.ssl_scale * ssl_loss + self.diff_scale * diff_loss + self.cls_scale * cls_loss,
                'clip_loss': clip_loss,
                'clip_acc': clip_acc,
                'ssl_loss': ssl_loss,
                'ssl_acc': ssl_acc,
                'diff_loss': diff_loss,
                'cls_loss': cls_loss}