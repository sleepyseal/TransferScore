import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from utils import proxy_a_distance, MMD_loss, uniformity
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier
from tllib.modules.kernels import GaussianKernel
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
import sys, os
from random import sample
from numpy.random import uniform
from sklearn.neighbors import NearestNeighbors
import numpy as np 
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    print(len(train_source_dataset))
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=not args.non_linear
    )

    # resume from the best checkpoint
    if args.phase != 'train' and args.phase != 'evaluation':
        checkpoint = torch.load('source.pt')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        print(classifier)
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature, source_label = collect_feature(train_source_loader, feature_extractor, device, max_num_features=100)
        target_feature, target_label = collect_feature(train_target_loader, feature_extractor, device, max_num_features=100)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, source_feature,source_label, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)

        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    if args.phase =='evaluation':
        get_transfer_score(train_target_iter, classifier, num_classes)
        return
    # start training
    best_acc1 = 0.

    for epoch in range(args.epochs):
        # train for one epoch
        save_path=osp.join('dan_weight', args.data, args.source[0]+'2'+args.target[0]+'_'+str(args.seed))
        os.makedirs(save_path, exist_ok=True)

        train(train_source_iter, train_target_iter, classifier, mkmmd_loss, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), osp.join(save_path, 'epoch'+ str(epoch)+'_'+str(round(acc1,3))+'.pt'))
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          mkmmd_loss: MultipleKernelMaximumMeanDiscrepancy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    mkmmd_loss.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = mkmmd_loss(f_s, f_t)
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def get_epoch_num(file):
    sub=file.split('_')[0]
    num=sub[5:]
    return int(num)

def get_transfer_score(train_target_iter, classifier, num_classes):
    ACC=[]
    Score_epoch=[]
    weight_path= osp.join('dan_weight', args.data, args.source[0]+'2'+args.target[0]+'_'+str(args.seed))
    weights=os.listdir(weight_path)
    weights=sorted(weights, key= get_epoch_num)

    for weight in weights:
        print(weight.split('.pt')[0].split('_')[-1])
        ACC.append(weight.split('.pt')[0].split('_')[1])
    acc_name='dan_acc_'+str(args.seed)+'.txt'
    with open(acc_name, 'w') as file:
        for a in ACC:
            file.write(str(a)+'\n')

    for weight in weights:
        if 'epoch' in weight:

            save_path=osp.join(weight_path, weight)
            classifier.load_state_dict(torch.load(save_path))

            im_loss=[]
            features=torch.tensor([0]).cuda().float()

            classifier.eval()
            u=0
            for k, v in classifier.head.named_parameters(): 
                if "weight" in k:
                    u=uniformity(v)
            iter_num=len(train_target_iter)
            with torch.no_grad():
                for i in range(iter_num):
                    x_t, = next(train_target_iter)[:1]
                    x_t = x_t.to(device)

                    y, output_f = classifier(x_t, require_feature=True)

                    if features.size()[0]==1:
                        features=output_f
                    else:
                        features= torch.cat((features, output_f), dim=0)

                    output_test=y

                    softmax_out = nn.Softmax(dim=1)(output_test)
                    entropy_loss = torch.mean(entropy(softmax_out))
                    # print(entropy_loss)

                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
                    entropy_loss -= gentropy_loss

                    im_loss.append(entropy_loss.item())

            X=features.cpu().numpy()

            sample_size = int(X.shape[0]*0.05) #0.05 (5%) based on paper by Lawson and Jures

            #a uniform random sample in the original data space
            X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
            random_indices=sample(range(0, X.shape[0], 1), sample_size)
            X_sample = X[random_indices]
 
            #initialise unsupervised learner for implementing neighbor searches
            neigh = NearestNeighbors(n_neighbors=2)
            nbrs=neigh.fit(X)
            
            u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
            u_distances = u_distances[: , 0] 
            
            w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
            w_distances = w_distances[: , 1]
            
            u_sum = np.sum(u_distances)
            w_sum = np.sum(w_distances)
            H = u_sum/ (u_sum + w_sum)

            score= H-sum(im_loss)/len(im_loss)/math.log(num_classes)-u
            Score_epoch.append(score)

    score_name='dan_score_'+str(args.seed)+'.txt'
    with open(score_name, 'w') as file:
        for s in Score_epoch:
            file.write(str(s)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--non-linear', default=False, action='store_true',
                        help='whether not use the linear version')
    parser.add_argument('--trade-off', default=1.0, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis', 'evaluation'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
