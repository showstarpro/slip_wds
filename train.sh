# python run_with_submitit.py \
#   --nodes 1 \
#   --ngpus 8 \
#   --batch-size 4 \
#   --train-data '/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar' \
#   --train-num-samples 10_968_539 \
#   --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
#   --model SLIP_VITB16 \
#   --lr 3e-3 --wd 0.1

export CUDA_VISIBLE_DEVICES=4,6,7

# torchrun --nproc_per_node 3 --master_port 12345  -m  main_wds \
#   --batch-size 4 \
#   --train-data '/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar' \
#   --train-num-samples 10_968_539 \
#   --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
#   --model CLIP_VITB16 \
#   --lr 1e-3 --wd 0.1 \
#   --output-dir /lpai/SLIP/logs

torchrun --nproc_per_node 3 --master_port 12345  -m  main_wds \
  --batch-size 4 \
  --train-data '/lpai/dataset/cc3m-3long-3short-1raw/0-1-0/cc3m_3long_3short_1raw_captions/00{000..287}.tar' \
  --train-num-samples  1711097 \
  --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
  --model CLIP_VITB16 \
  --lr 1e-3 --wd 0.1 \
  --output-dir /lpai/SLIP/logs