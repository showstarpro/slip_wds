# python run_with_submitit.py \
#   --nodes 1 \
#   --ngpus 8 \
#   --batch-size 4 \
#   --train-data '/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar' \
#   --train-num-samples 10_968_539 \
#   --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
#   --model SLIP_VITB16 \
#   --lr 3e-3 --wd 0.1

torchrun --nproc_per_node 8 --master_port 12345  -m  main_wds \
  --batch-size 16 \
  --train-data '/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar' \
  --train-num-samples 10_968_539 \
  --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
  --model SLIP_VITB16 \
  --lr 3e-3 --wd 0.1 \
  --output-dir /lpai/SLIP/logs