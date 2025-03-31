# torchrun --nproc_per_node 8 --master_port 12345  -m  main_linear \
#   --arch vit_base_patch16_224 --dataset imagenet \
#   --lr 0.01 \
#   --epochs 100 \
#   --pretrained /lpai/slip_base/SLIP/slip_base_cc12m_35ep.pt \
#   --output-dir /lpai/volumes/so-volume-ga/lhp/slip/linear_slip
#!/bin/bash
source /root/anaconda3/etc/profile.d/conda.sh 

conda activate superclass

BS=128

# CHKPNT=/lpai/models/slip/3morigin30e/origin_slip_cc3m_30ep/checkpoint_best.pt
CHKPNT=/lpai/models/slip/12morifin30e/origin_slip_cc12_30ep/checkpoint_best.pt

# CIFAR10=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar10
# CIFAR100=/lpai/volumes/so-volume-bd-ga/lhp/datasets/cifar100
# IMAGENETVAL=/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val
# IMAGENETTRAIN=/lpai/dataset/imagenet-1k/0-1-0/train
# FLOWERS102=/lpai/volumes/so-volume-bd-ga/lhp/datasets/flower_102/dataset
# FOOD101=/lpai/volumes/so-volume-bd-ga/lhp/datasets/food_101
# STANFORD=/lpai/volumes/so-volume-bd-ga/lhp/datasets

# CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 torchrun --nproc_per_node=6 -m main_linear --arch vit_base_patch16_224 --batch-size=$BS --workers=2 --dataset cifar10 --pretrained $CHKPNT --num-classes 10
# CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 torchrun --nproc_per_node=6 -m main_linear --arch vit_base_patch16_224 --batch-size=$BS --workers=2 --dataset cifar100 --pretrained $CHKPNT --num-classes 100
torchrun --nproc_per_node=8 -m main_linear --arch vit_base_patch16_224 --batch-size=$BS --workers=2 --dataset imagenet --pretrained $CHKPNT --num-classes 1000
# CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 torchrun --nproc_per_node=6 -m main_linear --arch vit_base_patch16_224 --batch-size=$BS --workers=2 --dataset flowers102 --pretrained $CHKPNT --num-classes 102
# CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 torchrun --nproc_per_node=6 -m main_linear --arch vit_base_patch16_224 --batch-size=$BS --workers=2 --dataset stanfordcars --pretrained $CHKPNT --num-classes 196