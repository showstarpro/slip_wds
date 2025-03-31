BS=4096

CHKPNT=/lpai/models/slip/3morigin30e/origin_slip_cc3m_30ep/checkpoint_best.pt

# CUDA_VISIBLE_DEVICES=6 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d imagenet-a
# CUDA_VISIBLE_DEVICES=6 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d imagenet-r
CUDA_VISIBLE_DEVICES=6 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d imagenet-sketch