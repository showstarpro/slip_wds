#!/bin/bash

BS=4096

# CHKPNT=/lpai/models/slip/3morigin30e/origin_slip_cc3m_30ep/checkpoint_best.pt
CHKPNT=/lpai/models/slip/12morifin30e/origin_slip_cc12_30ep/checkpoint_best.pt

# CIFAR10=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar10
# CIFAR100=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar100
# IMAGENETVAL=/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val
# IMAGENETTRAIN=/lpai/dataset/imagenet-1k/0-1-0/train
# FLOWERS102=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flower_102/dataset
# FOOD101=/lpai/volumes/so-volume-bd-ga/lhp/datasets/food_101
# STANFORD=/lpai/volumes/so-volume-bd-ga/lhp/datasets

# CUDA_VISIBLE_DEVICES=3 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d imagenet
# CUDA_VISIBLE_DEVICES=3 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d cifar10
CUDA_VISIBLE_DEVICES=3 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d cifar100
# CUDA_VISIBLE_DEVICES=4 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d flowers102
# CUDA_VISIBLE_DEVICES=6 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d stanfordcars