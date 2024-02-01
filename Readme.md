Can We Evaluate Domain Adaptation Models Without Target-Domain Labels?  
============================================================================
This repo is the official implementation of our paper
*"[Can We Evaluate Domain Adaptation Models Without Target-Domain Labels?](https://openreview.net/forum?id=fszrlQ2DuP)"*
Accepted by **ICLR2024** , to cite this work:
```
@inproceedings{yang2023can,
  title = {Can We Evaluate Domain Adaptation Models Without Target-Domain Labels?},
  author = {Yang, Jianfei and Qian, Hanjie and Xu, Yuecong and Wang, Kai and Xie, Lihua},
  booktitle = {International Conference on Learning Representations},
  Month = {May},
  year = {2024}
}
```
## Environment
1. Install `pytorch` and `torchvision` (we use pytorch==1.8.1 and torchvision==0.8.2).
2. `pip install -r requirements.txt`

## Datasets

Following datasets can be downloaded automatically into data folder:
```
Transfer-Score/
├── data/
    ├── office_home/
    │   ├── Art
    │   ├── Clipart
    │   ├── Product
    │   ├── Real World
    ├── office31/
    │   ├── amazon
    │   ├── dslr
    │   ├── webcam
    ├── visda17/
    │   ├── train
    │   ├── validation 
    ├── DomainNet/
    │   ├── ${DN_domain}/
    │   ├── ${DN_domain}_train.txt
    │   ├── ${DN_domain}_test.txt
```
## Demo Train on Office31 and evaluate the checkpoint

To train a UDA method on Office31 D2A and save the model:
```
python train.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2A
```
To calculate the transfer score of each checkpoint on Office31 D2A:
```
python train.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --log logs/dan/Office31_D2A --phase evaluation
```
## Demo Train on Office31 with different hyper-parameter (learning rate)
To train a UDA method on Office31 D2A with different learning rate and save the model:
```
python train.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 0 --lr 1 --log logs/dan/Office31_D2A
```
Then calculate the transfer score

## To calculate the transfer score
```
transfer_score=get_transfer_score(train_target_iter, classifier, num_classes)
```
- train_target_iter: the dataloader of target data  
- classifier: the model after UDA  
- num_classes: the number of classes
  
## Acknowledge
We would like to thank https://github.com/thuml/Transfer-Learning-Library
