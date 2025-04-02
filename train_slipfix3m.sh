source /root/anaconda3/etc/profile.d/conda.sh 

conda activate superclass

torchrun --nproc_per_node 8 --master_port 12345  -m  main_wds \
  --batch-size 256 \
  --train-data '/lpai/dataset/cc3m-webdataset/0-1-0/cc3m/cc3m-train-{0000..0575}.tar' \
  --train-num-samples 2905954 \
  --imagenet-val '/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' \
  --model SLIP_VITB16 \
  --lr 3e-3 --wd 0.1 \
  --output-dir /lpai/output/models