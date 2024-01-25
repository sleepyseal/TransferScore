# #!/usr/bin/env bash
python train.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2A
python train.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2A --phase evaluation

