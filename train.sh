# python run_with_submitit.py \
#   --nodes 1 \
#   --ngpus 8 \
#   --batch-size 4 \
#   --train-data '/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar' \
#   --train-num-samples 10_968_539 \
#   --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
#   --model SLIP_VITB16 \
#   --lr 3e-3 --wd 0.1
# 00014111  00013957 00013788 00014070 00013974 00014013 00013938 00013972 00013992 00013993 00000000
# 1            10       2      3         4         5        6        7      8        9
# # 这个假设是 data_dir 部分是可以按某种规则连续编号或使用特定范围的
# /lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir{1..10}/shards/shard-{00000000..00014111}.tar
# '/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir1/shards/{00000000..00014111}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir10/shards/{00000000..00013957}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir2/shards/{00000000..00013788}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir3/shards/{00000000..00014070}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir4/shards/{00000000..00013974}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir5/shards/{00000000..00014013}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir6/shards/{00000000..00013938}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir7/shards/{00000000..00013972}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir8/shards/{00000000..00013992}.tar::/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir9/shards/{00000000..00013993}.tar'
# '/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir*/shards/*.tar'
# /lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir*/shards/*.tar

torchrun --nproc_per_node 8 --master_port 12345  -m  main_wds \
  --batch-size 64 \
  --train-data '/lpai/dataset/datacomp1b/0-2-0/datacomp1b_finished/data_dir*/shards/*.tar' \
  --train-num-samples 10_968_539 \
  --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
  --model CLIP_VITB16 \
  --lr 3e-3 --wd 0.1 \
  --output-dir /lpai/SLIP/logs