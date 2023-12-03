# #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2A
# CUDA_VISIBLE_DEVICES=0 python dan.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2A --phase evaluation

