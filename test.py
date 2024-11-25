import torch
from tokenizer import SimpleTokenizer
from data import get_data
import torchvision.transforms as transforms
import argparse

# tokenizer = SimpleTokenizer()

# texts = "我爱吃大米$"

# txt_token = tokenizer(texts)

# print(txt_token)

# txts = tokenizer.decode(txt_token.cpu().numpy())
# print(txts)

dataset_path = '/lpai/dataset/datacomp-13m/0-1-0/datacomp_small/shards/{00000000..00001287}.tar' 

tokenizer = SimpleTokenizer()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
preprocess_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        normalize
    ])
preprocess_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])



# parser = argparse.ArgumentParser(description='SLIP training and evaluation', add_help=False)
# # Data

# parser.add_argument('--train_data', default='/lpai/dataset/datacomp-13m/0-1-0/datacomp_small/shards/{00000000..00001287}.tar' ,type=str)
# parser.add_argument('--val_data', type=str)
# parser.add_argument('--imagenet_val',default='/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val' ,type=str)
# parser.add_argument('--imagenet_v2', type=str)
# parser.add_argument('--dataset_type',default="webdataset" ,type=str)

# parser.add_argument('--dataset', default='yfcc15m', type=str, choices=['yfcc15m', 'cc3m', 'cc12m', 'coco', 'redcaps'])
# parser.add_argument('--root', default='', type=str,
#                     help='path to dataset root')
# parser.add_argument('--metadata', default='yfcc15m.pkl', type=str,
#                     help='path to metadata file (see README for details)')
# parser.add_argument('--output-dir', default='./', type=str, help='output dir')
# # Model
# parser.add_argument('--model', default='SLIP_VITB16', type=str)
# parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
#                     help='hidden dim of SimCLR mlp projection head')
# parser.add_argument('--ssl-emb-dim', default=256, type=int,
#                     help='output embed dim of SimCLR mlp projection head')
# parser.add_argument('--ssl-scale', default=1.0, type=float,
#                     help='loss scale for SimCLR objective')
# parser.add_argument('--ssl-temp', default=0.1, type=float,
#                     help='softmax temperature for SimCLR objective')
# parser.add_argument('--resume', default='', type=str, help='path to resume from')
# # Training
# parser.add_argument('--epochs', default=1, type=int)
# parser.add_argument('--warmup-epochs', default=1, type=int)
# parser.add_argument('--start-epoch', default=0, type=int)
# parser.add_argument('--batch-size', default=64, type=int,
#                     help='number of samples per-device/per-gpu')
# parser.add_argument('--lr', default=3e-3, type=float)
# parser.add_argument('--lr-start', default=1e-6, type=float,
#                     help='initial warmup lr')
# parser.add_argument('--lr-end', default=1e-5, type=float,
#                     help='minimum final lr')
# parser.add_argument('--update-freq', default=1, type=int,
#                     help='optimizer update frequency (i.e. gradient accumulation steps)')
# parser.add_argument('--wd', default=0.1, type=float)
# parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
# parser.add_argument('--eps', default=1e-8, type=float)
# parser.add_argument('--eval-freq', default=1, type=int)
# parser.add_argument('--disable-amp', action='store_true',
#                     help='disable mixed-precision training (requires more memory and compute)')
# # System
# parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
# parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
#                     help='number of data loading workers per process')
# parser.add_argument('--evaluate', action='store_true', help='eval only')
# parser.add_argument('--world-size', default=1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=0, type=int,
#                     help='node rank for distributed training')
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument('--dist-url', default='env://', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str)
# parser.add_argument('--seed', default=0, type=int)
# parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
# parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')


# args = parser.parse_args()

# args.train_num_samples = 128_000_000
# args.train_data_upsampling_factors = None

device = "cuda:0"

parser = argparse.ArgumentParser(description='SLIP training and evaluation', add_help=False)
# Data

parser.add_argument(
        "--train-data",
        type=str,
        default='/lpai/dataset/cc12m/0-1-0/cc12m-wds/cc12m-train-{0000..2175}.tar',
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
parser.add_argument(
    "--train-data-upsampling-factors",
    type=str,
    default=None,
    help=(
        "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
        "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
        "By default, datapoints are sampled uniformly regardless of the dataset sizes."
    )
)
parser.add_argument(
    "--val-data",
    type=str,
    default=None,
    help="Path to file(s) with validation data",
)
parser.add_argument(
    "--train-num-samples",
    type=int,
    default=10_968_539,
    help="Number of samples in dataset. Required for webdataset if not available in info file.",
)
parser.add_argument(
    "--val-num-samples",
    type=int,
    default=None,
    help="Number of samples in dataset. Useful for webdataset if not available in info file.",
)
parser.add_argument(
    "--dataset-type",
    choices=["webdataset", "csv", "synthetic", "auto"],
    default="webdataset",
    help="Which type of dataset to process."
)
parser.add_argument(
    "--dataset-resampled",
    default=False,
    action="store_true",
    help="Whether to use sampling with replacement for webdataset shard selection."
)
parser.add_argument(
    "--csv-separator",
    type=str,
    default="\t",
    help="For csv-like datasets, which separator to use."
)
parser.add_argument(
    "--csv-img-key",
    type=str,
    default="filepath",
    help="For csv-like datasets, the name of the key for the image paths."
)
parser.add_argument(
    "--csv-caption-key",
    type=str,
    default="title",
    help="For csv-like datasets, the name of the key for the captions."
)
parser.add_argument(
    "--imagenet-val",
    type=str,
    default='/lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val',
    help="Path to imagenet val set for conducting zero shot evaluation.",
)
parser.add_argument(
    "--imagenet-v2",
    type=str,
    default=None,
    help="Path to imagenet v2 for conducting zero shot evaluation.",
)
parser.add_argument(
    "--cache-dir",
    type=str,
    default=None,
    help="Override system default cache path for model & tokenizer file downloads.",
)
# Model
parser.add_argument('--model', default='SLIP_VITB16', type=str)
parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                    help='hidden dim of SimCLR mlp projection head')
parser.add_argument('--ssl-emb-dim', default=256, type=int,
                    help='output embed dim of SimCLR mlp projection head')
parser.add_argument('--ssl-scale', default=1.0, type=float,
                    help='loss scale for SimCLR objective')
parser.add_argument('--ssl-temp', default=0.1, type=float,
                    help='softmax temperature for SimCLR objective')
parser.add_argument('--resume', default='', type=str, help='path to resume from')
# Training
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--warmup-epochs', default=1, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--batch-size', default=64, type=int,
                    help='number of samples per-device/per-gpu')
parser.add_argument('--lr', default=3e-3, type=float)
parser.add_argument('--lr-start', default=1e-6, type=float,
                    help='initial warmup lr')
parser.add_argument('--lr-end', default=1e-5, type=float,
                    help='minimum final lr')
parser.add_argument('--update-freq', default=1, type=int,
                    help='optimizer update frequency (i.e. gradient accumulation steps)')
parser.add_argument('--wd', default=0.1, type=float)
parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--eval-freq', default=1, type=int)
parser.add_argument('--disable-amp', action='store_true',
                    help='disable mixed-precision training (requires more memory and compute)')
# System
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers per process')
parser.add_argument('--evaluate', action='store_true', help='eval only')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')


args = parser.parse_args()

data = get_data(
    args,
    (preprocess_train, preprocess_val),
    epoch=0,
    tokenizer=tokenizer,
)


data['train'].set_epoch(0)  # set epoch in process safe manner via sampler or shared_epoch
dataloader = data['train'].dataloader

val_loader = data['imagenet-val'].dataloader

for i, (images, target) in enumerate(val_loader):
    print(type(images))
    print(type(target))
# 遍历数据加载器
for batch in dataloader:
    imgs1, imgs2, texts = batch
    print(imgs2.size())