BS=128

# CHKPNT=/lpai/models/slip/3morigin30e/origin_slip_cc3m_30ep/checkpoint_best.pt
CHKPNT=/lpai/models/slip/12morifin30e/origin_slip_cc12_30ep/checkpoint_best.pt

# CUDA_VISIBLE_DEVICES=6 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d mscoco --ms-coco
CUDA_VISIBLE_DEVICES=6 python -m zeroshot --resume $CHKPNT --batch-size=$BS --workers=2 --d flickr30k --flickr