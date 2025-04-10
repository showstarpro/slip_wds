# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch import distributed as dist
import utils


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = {}
        self.last_local_batch_size = None
        # self.mask = None
    def get_ground_truth(self, local_batch_size, device):
        if local_batch_size != self.last_local_batch_size or device not in self.labels:
            labels = torch.arange(
                local_batch_size, device=device
            )
            # total_batch_size = local_batch_size * utils.get_world_size()
            # self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.labels[device] = labels
            self.last_local_batch_size = local_batch_size
        else:
            labels = self.labels[device]
        return labels
    
    def gather_features(self, image1_features, image2_features, text1_features, text2_features, sentence1_features, sentence2_features):
        world_size = utils.get_world_size()
        rank = utils.get_rank()
        gathered_image1_features = [torch.zeros_like(image1_features) for _ in range(world_size)]
        gathered_image2_features = [torch.zeros_like(image2_features) for _ in range(world_size)]
        gathered_text1_features = [torch.zeros_like(text1_features) for _ in range(world_size)]
        gathered_text2_features = [torch.zeros_like(text2_features) for _ in range(world_size)]
        gathered_sentence1_features = [torch.zeros_like(sentence1_features) for _ in range(world_size)]
        gathered_sentence2_features = [torch.zeros_like(sentence2_features) for _ in range(world_size)]
        dist.all_gather(gathered_image1_features, image1_features)
        dist.all_gather(gathered_image2_features, image2_features)
        dist.all_gather(gathered_text1_features, text1_features)
        dist.all_gather(gathered_text2_features, text2_features)
        dist.all_gather(gathered_sentence1_features, sentence1_features)
        dist.all_gather(gathered_sentence2_features, sentence2_features)
        # ensure grads for local rank when all_* features don't have a gradient
        gathered_image1_features[rank] = image1_features
        gathered_image2_features[rank] = image2_features
        gathered_text1_features[rank] = text1_features
        gathered_text2_features[rank] = text2_features
        gathered_sentence1_features[rank] = sentence1_features
        gathered_sentence2_features[rank] = sentence2_features
        all_image1_features = torch.cat(gathered_image1_features, dim=0)
        all_image2_features = torch.cat(gathered_image2_features, dim=0)
        all_text1_features = torch.cat(gathered_text1_features, dim=0)
        all_text2_features = torch.cat(gathered_text2_features, dim=0)
        all_sentence1_features = torch.cat(gathered_sentence1_features, dim=0)
        all_sentence2_features = torch.cat(gathered_sentence2_features, dim=0)

        return all_image1_features, all_image2_features, all_text1_features, all_text2_features, all_sentence1_features, all_sentence2_features

    def forward(self, outputs):
        image1_embed = outputs['image1_embed']
        image2_embed = outputs['image2_embed']
        text1_embed = outputs['text1_embed']
        text2_embed = outputs['text2_embed']
        sentence1_features = outputs['sentence1_features']
        sentence2_features = outputs['sentence2_features']
        logit_scale = outputs['logit_scale']

        # normalized features
        image1_embed = F.normalize(image1_embed, dim=-1, p=2)
        image2_embed = F.normalize(image2_embed, dim=-1, p=2)
        text1_embed = F.normalize(text1_embed, dim=-1, p=2)
        text2_embed = F.normalize(text2_embed, dim=-1, p=2)

        ###-----###
        image1_embed_all, image2_embed_all, text1_embed_all, text2_embed_all, all_sentence1_features, all_sentence2_features = self.gather_features(
            image1_features=image1_embed, image2_features=image2_embed, text1_features=text1_embed, text2_features=text2_embed, sentence1_features=sentence1_features, sentence2_features=sentence2_features
        )
        ###=----###
        # gather features from all GPUs
        # image1_embed_all, text_embed_all = \
        #     utils.all_gather_batch([image1_embed, text_embed])

        # cosine similarity as logits
        logits_per_image1 = logit_scale * image1_embed_all @ text2_embed_all.t()
        logits_per_text1 = logit_scale * text2_embed_all @ image1_embed_all.t()

        local_batch_size = logits_per_image1.size(0)
        device = image1_embed.device
        labels = self.get_ground_truth(local_batch_size, device)
        
        loss1 = (F.cross_entropy(logits_per_image1, labels) + \
            F.cross_entropy(logits_per_text1, labels)) / 2

        # gather features from all GPUs
        # image2_embed_all, text_embed_all = \
        #     utils.all_gather_batch([image2_embed, text_embed])

        # cosine similarity as logits
        logits_per_image2 = logit_scale * image2_embed_all @ text1_embed_all.t()
        logits_per_text2 = logit_scale * text1_embed_all @ image2_embed_all.t()

        loss2 = (F.cross_entropy(logits_per_image2, labels) + \
            F.cross_entropy(logits_per_text2, labels)) / 2

        clip_loss = (loss1 + loss2) /2


        #### ------------------------- ####
        #### add for sentence_image_aug ####
        #### ------------------------- ####
        # gather features from all GPUs
        # all_sentence1_features, all_sentence2_features = \
        #     utils.all_gather_batch([sentence1_features, sentence2_features])
        
        logits_per_sentence11 = logit_scale * all_sentence1_features @ all_sentence1_features.T
        logits_per_sentence11 = logits_per_sentence11 - F.one_hot(labels, logits_per_sentence11.shape[0]) * 1e9
        logits_per_sentence22 = logit_scale * all_sentence2_features @ all_sentence2_features.T
        logits_per_sentence22 = logits_per_sentence22 - F.one_hot(labels, logits_per_sentence22.shape[0]) * 1e9
        
        logits_per_sentence12 = logit_scale * all_sentence1_features @ all_sentence2_features.T
        logits_per_sentence21 = logit_scale * all_sentence2_features @ all_sentence1_features.T

        loss_sentence1 = F.cross_entropy(torch.cat([logits_per_sentence12, logits_per_sentence11], dim=1), labels)
        loss_sentence2 = F.cross_entropy(torch.cat([logits_per_sentence21, logits_per_sentence22], dim=1), labels)

        sentence_loss =  2 * (loss_sentence1 + loss_sentence2) / 2

        loss = clip_loss + sentence_loss
        #### ------------------------- ####


        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image1, dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_loss': clip_loss, 'sentence_loss': sentence_loss, 'clip_acc': acc}


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
        clip_loss = clip_loss_dict['loss']
        clip_acc = clip_loss_dict['clip_acc']

        ssl_loss_dict = self.ssl_loss(outputs)
        ssl_loss = ssl_loss_dict['ssl_loss']
        ssl_acc = ssl_loss_dict['ssl_acc']

        return {'loss': clip_loss + self.ssl_scale * ssl_loss,
                'clip_loss': clip_loss,
                'sentence_loss': clip_loss_dict['sentence_loss'],
                'clip_acc': clip_acc,
                'ssl_loss': ssl_loss,
                'ssl_acc': ssl_acc}
