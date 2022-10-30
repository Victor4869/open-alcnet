from fileinput import close
import os
# os.system("taskset -p -c 40-47 %d" % os.getpid())

import platform
import sys
import socket
import argparse
from xmlrpc.client import boolean
import numpy as np
from tqdm import tqdm

from datetime import datetime

import mxnet as mx
from mxnet import gluon, autograd, init
from mxnet.gluon.data.vision import transforms

from gluoncv.utils import LRScheduler

from data import IceContrast
from model import MPCMResNetFPN
from metric import SigmoidMetric, SamplewiseSigmoidMetric
from loss import SoftIoULoss, SamplewiseSoftIoULoss

import matplotlib.pyplot as plt

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')

    # host information
    parser.add_argument('--host', type=str, default='xxx',
                        help='xxx is a place holder, leave as default or replace with \
                            your host information. e.g. host name or GPU details')

    # model
    parser.add_argument('--net-choice', type=str, default='PCMNet',
                        help='model name PCMNet, PlainNet')
    parser.add_argument('--pyramid-mode', type=str, default='Dec',
                        help='Inc, Dec')
    parser.add_argument('--r', type=int, default=2, help='1, 2, 4')
    parser.add_argument('--summary', action='store_true',
                        help='print parameters')
    parser.add_argument('--scale-mode', type=str, default='xxx',
                        help='Single, Multiple, Selective')
    parser.add_argument('--pyramid-fuse', type=str, default='sk',
                        help='add, max, sk, bottomuplocal')
    parser.add_argument('--cue', type=str, default='lcm', help='lcm or orig')
    # dataset
    parser.add_argument('--dataset', type=str, default='DENTIST',
                        help='folder name of your dataset (default: DENTIST, Iceberg)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='use this if your dataset is outside the alcnet folder, \
                            enter the root path of your dataset folder (default: None)')
    parser.add_argument('--workers', type=int, default=48,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--blocks', type=int, default=4,
                        help='[1] * blocks')
    parser.add_argument('--channels', type=int, default=16,
                        help='channels')
    parser.add_argument('--shift', type=int, default=13,
                        help='shift')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='iou-thresh')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', type=str, default='val',
                        help='dataset val split (default: val)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=32,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,200',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--gamma', type=int, default=2,
                        help='gamma for Focal Soft IoU Loss')
    parser.add_argument('--lambd', type=int, default=1,
                        help='lambd for TV Soft IoU Loss')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    parser.add_argument('--sparsity', action='store_true', default=
                        False, help='')
    parser.add_argument('--score-thresh', type=float, default=0.5,
                        help='score-thresh')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # parser.add_argument('--no-cuda', action='store_true', default=
    #                     True, help='disables CUDA training')
    # parser.add_argument('--ngpus', type=int,
    #                     default=len(mx.test_utils.list_gpus()),
    #                     help='number of GPUs (default: 4)')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    # using colab                    
    parser.add_argument('--colab', action='store_true', default=
                        False, help='whether using colab')
    parser.add_argument('--colab-path', type=str, default=None,
                        help='put the path of the ALCNet folder in Colab')

    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    parser.add_argument('--metric', type=str, default='IoU',
                        help='F1, IoU, mAP')

    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda or (len(mx.test_utils.list_gpus()) == 0):
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        # print('Number of GPUs:', args.ngpus)
        # args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
        args.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        print('Number of GPUs:', len(args.ctx))

    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    # args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    args.norm_kwargs = {'num_devices': len(args.ctx)} if args.syncbn else {}
    # print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        # print arguments in multiple lines
        tmp = str(args).split(",")
        self.arg_string = ""
        line = ''
        for k in tmp:
            if len(line + k) > 90:
                self.arg_string = self.arg_string + '\n' + k[1:] + ','
                line = k
            else:
                self.arg_string = self.arg_string + k + ','
                line = line + k
        self.arg_string = self.arg_string[:-1] + '\n'
        print (self.arg_string)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]), # Default mean and std
            # transforms.Normalize([.418, .447, .571], [.091, .078, .076]),   # Iceberg mean and std
        ])
        ################################# dataset and dataloader #################################
# =============================================================================
#         if platform.system() == "Darwin":
#             data_root = os.path.join('~', 'Nutstore Files', 'Dataset')
#         elif platform.system() == "Linux":
#             data_root = os.path.join('~', 'datasets')
#             if args.colab:
#                 # data_root = '/content/gdrive/My Drive/Colab Notebooks/datasets'
#                 data_root = '/content/datasets'
#         elif platform.system() == "Windows":
#             data_root = os.path.join(os.path.abspath(os.getcwd()), 'alcnet', 'datasets')
#         else:
#             raise ValueError('Notice Dataset Path')
# =============================================================================


        # get the path of alcnet folder
        if args.colab:
            if args.colab_path is not None:
                self.alc_dir = args.colab_path
            else:
                raise RuntimeError("Plese enter the full path of ALCNet in Colab to argument --colab-path")
        else:
            self.alc_dir = os.getcwd()
        
        # get the root path of the dataset
        if args.data_root is not None:
            self.data_root = args.data_root
        else:
            self.data_root = self.alc_dir

        data_kwargs = {'base_size': args.base_size, 'transform': input_transform,
                       'crop_size': args.crop_size, 'root': self.data_root,
                       'base_dir' : args.dataset}
        trainset = IceContrast(split=args.train_split, mode='train',   **data_kwargs)
        valset = IceContrast(split=args.val_split, mode='testval', **data_kwargs)

        self.train_data = gluon.data.DataLoader(trainset, args.batch_size, shuffle=True,
            last_batch='rollover', num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='rollover', num_workers=args.workers)

        # net_choice = 'PCMNet'  # ResNetFPN, PCMNet, MPCMNet, LayerwiseMPCMNet
        net_choice = self.args.net_choice
        print("net_choice: ", net_choice)

        model = '' #Fix scope issue

        if net_choice == 'MPCMResNetFPN':
            r = self.args.r
            layers = [self.args.blocks] * 3
            channels = [8, 16, 32, 64]
            shift = self.args.shift
            pyramid_mode = self.args.pyramid_mode
            scale_mode = self.args.scale_mode
            pyramid_fuse = self.args.pyramid_fuse

            model = MPCMResNetFPN(layers=layers, channels=channels, shift=shift,
                                  pyramid_mode=pyramid_mode, scale_mode=scale_mode,
                                  pyramid_fuse=pyramid_fuse, r=r, classes=trainset.NUM_CLASS)
            # print("net_choice: ", net_choice)
            print("scale_mode: ", scale_mode)
            print("pyramid_fuse: ", pyramid_fuse)
            print("r: ", r)
            print("layers: ", layers)
            print("channels: ", channels)
            print("shift: ", shift)

        if args.host == 'xxx':
            self.host_name = socket.gethostname()  # automatic
        else:
            self.host_name = args.host             # Your desired host information

        self.save_prefix = self.host_name + '_' + net_choice + '_' + args.scale_mode + \
                           '_' + args.pyramid_fuse + '_r_' + str(args.r) + '_b_' + str(args.blocks)
        if args.net_choice == 'ResNetFCN':
            self.save_prefix = self.host_name + '_' + net_choice + '_b_' + str(args.blocks)

        # resume checkpoint if needed
        if args.resume is not None:
            args.resume = os.path.expanduser(args.resume)
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
                print("Model resumed")
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            # model.initialize(init=init.Xavier(), ctx=args.ctx, force_reinit=True)
            model.initialize(init=init.MSRAPrelu(), ctx=args.ctx, force_reinit=True)
            print("Model Initializing")
            print("args.ctx: ", args.ctx)


        self.net = model
        # self.net.summary(mx.nd.zeros((1, 3, 480, 480)))

        if args.summary:
            self.net.summary(mx.nd.zeros((1, 3, 480, 480), self.args.ctx[0]))
            sys.exit()

        # create criterion
        self.criterion = SoftIoULoss()

        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr,
                                        nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_data),
                                        power=0.9)
        kv = mx.kv.create(args.kvstore)

        # For SGD
        # optimizer_params = {'lr_scheduler': self.lr_scheduler,
        #                     'wd': args.weight_decay,
        #                     'momentum': args.momentum,
        #                     'learning_rate': args.lr
        #                    }
        optimizer_params = {
            # 'lr_scheduler': self.lr_scheduler,
            'wd': args.weight_decay,
            'learning_rate': args.lr
        }
        # For Adam

        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = Truedate_string

        if args.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        # self.optimizer = gluon.Trainer(self.net.collect_params(), 'sgd',
        #                                optimizer_params, kvstore = kv)
        # self.optimizer = gluon.Trainer(self.net.collect_params(), 'adam',
        #                                optimizer_params, kvstore = kv)
        self.optimizer = gluon.Trainer(self.net.collect_params(), 'adagrad',
                                       optimizer_params, kvstore=kv)
        # self.optimizer = gluon.Trainer(self.net.collect_params(), 'nag',
        #                                optimizer_params, kvstore=kv)

        ################################# evaluation metrics #################################

        self.iou_metric = SigmoidMetric(1)
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=self.args.score_thresh)
        # self.metric = Seg2DetVOC07MApMetric(iou_thresh=self.args.iou_thresh,
        #                                     sparsity=self.args.sparsity,
        #                                     score_thresh=self.args.score_thresh)
        self.best_metric = 0
        self.best_iou = 0
        self.best_nIoU = 0
        self.is_best = False

        self.train_losses = []
        self.val_losses = []
        self.nets = []
        self.ious = []
        self.nious = []

        

        date = datetime.now()
        self.date_string = date.strftime("%d-%m-%Y_") # date for log file name

        self.param_save_path = self.alc_dir + '/params/' + date.strftime("%d-%m-%Y_%H-%M-%S_") + self.host_name + "/"    # path to save parameter files
        # print(self.param_save_path)

        # make folder for log and parameter files
        try:
            os.makedirs(self.param_save_path)
            print(self.param_save_path + ' did not existed and was created.')
        except:
            print("Parameter folder already found: " + self.param_save_path)

        
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # time for log message

        # create log files
        if args.eval is False:
            # checkpoint log
            try:
                path = self.param_save_path + 'checkpoint/'
                os.makedirs(path)
                print(path + ' did not existed and was created.')
            except:
                print("checkpoint folder already found: " + self.param_save_path)

            with open(self.param_save_path + 'checkpoint/' + 'checkpoint.log', 'a') as f:
                f.write('\n{} {}\n'.format(dt_string, self.arg_string))
                f.write('Check point will be created when all the conditions are met: \n')
                f.write('1. Operates in tranining mode and current epoch > 40 \n')
                f.write('2. Validation loss increases and tranining loss decreases for 2 successive epochs.\n')
                f.write('3. Validation loss is higher than tranining loss. \n')

            # best IoU log    
            with open(self.param_save_path + 'best_IoU.log', 'a') as f:
                # first log message
                f.write('\n{} {}\n'.format(dt_string, self.arg_string))
                f.write('Training mode\n')
                if args.resume is not None:
                    args.resume = os.path.expanduser(args.resume)
                    if os.path.isfile(args.resume):
                        print("Model resumed")
                    else:
                        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

            # best nIoU log
            with open(self.param_save_path + 'best_nIoU.log', 'a') as f:
                # first log message
                f.write('\n{} {}\n'.format(dt_string, self.arg_string))
                f.write('Training mode\n')
                if args.resume is not None:
                    args.resume = os.path.expanduser(args.resume)
                    if os.path.isfile(args.resume):
                        print("Model resumed")
                    else:
                        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            # evaluation log    
            with open(self.param_save_path + 'evaluation.log', 'a') as f:
                # first log message
                f.write('\n{} {}\n'.format(dt_string, self.arg_string))
                f.write('Evaluation mode\n')
                if os.path.isfile(args.resume):
                    f.write("Model resumed\n")
                else:
                    raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        train_loss_ave = 0.0
        for i, batch in enumerate(tbar):
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.args.ctx, batch_axis=0)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=self.args.ctx, batch_axis=0)
            losses = []
            with autograd.record(True):
                for x, y in zip(data, labels):
                    pred = self.net(x)
                    loss = self.criterion(pred, y)
                    losses.append(loss)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            train_loss_ave = train_loss/(i+1)
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, train_loss_ave))
        self.train_losses.append(train_loss_ave)

        if args.eval is False:
            # save the model for each epoch, works in tranining mode only
            # unable to get deep copy directly from self.net
            # work around to get deep copy, save parameter file first then load
            self.net.save_parameters(self.param_save_path + 'last.params')
            last_model = ''
            if args.net_choice == 'MPCMResNetFPN':
                r = self.args.r
                layers = [self.args.blocks] * 3
                channels = [8, 16, 32, 64]
                shift = self.args.shift
                pyramid_mode = self.args.pyramid_mode
                scale_mode = self.args.scale_mode
                pyramid_fuse = self.args.pyramid_fuse

                last_model = MPCMResNetFPN(layers=layers, channels=channels, shift=shift,
                                    pyramid_mode=pyramid_mode, scale_mode=scale_mode,
                                    pyramid_fuse=pyramid_fuse, r=r)
            last_model.load_parameters(self.param_save_path + 'last.params', mx.cpu(0))
            self.nets.append(last_model)

    def validation(self, epoch):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        # self.metric.reset()
        tbar = tqdm(self.eval_data)
        fig = plt.figure()
        val_loss = 0.0
        val_loss_ave = 0.0
        for i, batch in enumerate(tbar):
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.args.ctx, batch_axis=0)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=self.args.ctx, batch_axis=0)
            preds = []
            losses = []
            for x, y in zip(data, labels):
                pred = self.net(x)
                preds.append(pred)
                loss = self.criterion(pred, y)
                losses.append(loss)

            for loss in losses:
                val_loss += np.mean(loss.asnumpy()) / len(losses)
            
            val_loss_ave = val_loss/(i+1)
            
            # self.metric.update(preds, labels)
            self.iou_metric.update(preds, labels)
            self.nIoU_metric.update(preds, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            tbar.set_description('Epoch %d, validation loss %.4f, IoU: %.4f, nIoU: %.4f' % (epoch, val_loss_ave, IoU, nIoU))
        self.val_losses.append(val_loss_ave)
        
        # training
        if args.eval is False:
            self.ious.append(IoU)
            self.nious.append(nIoU)

            # save the model if there is sign of overfitting
            if epoch > 40:
                if self.val_losses[-1] > self.val_losses[-2] and \
                    self.val_losses[-1] > self.val_losses[-3] and \
                    self.train_losses[-1] < self.train_losses[-2] and \
                    self.train_losses[-1] < self.train_losses[-3] and \
                    self.val_losses[-1] > self.train_losses[-1]:
                    self.nets[-3].save_parameters(self.param_save_path + 'checkpoint/' +'{}epoch.params'.format(epoch-2))
                    # log the check point information
                    with open(self.param_save_path + 'checkpoint/' + 'checkpoint.log', 'a') as f:
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                        f.write('{} - checkpoint saved at epoch: {:04d} Training loss: {:.4f} Validation loss: {:.4f} IoU: {:.4f} nIoU: {:.4f}\n'\
                                .format(dt_string, epoch-2, self.train_losses[-3], self.val_losses[-3],self.ious[-3], self.nious[-3]))
                    print("Sign of overfitting, saved checkpoint at {} epoch.".format(epoch-2))
            
            # Save plot for ious and losses every 5 epoch
            if epoch % 5 == 0:
                # for loss
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(range(epoch+1), self.train_losses)
                ax.plot(range(epoch+1), self.val_losses)
                ax.plot(range(epoch+1), self.ious)
                ax.plot(range(epoch+1), self.nious)
                ax.legend(["Train loss","Val loss","IoU","nIoU"])
                ax.set_xlabel("Epoch")
                fig.savefig(self.param_save_path + "IoU_losses.png")
                plt.close('all')

            if IoU > self.best_iou:
                self.best_iou = IoU

                # save the best model
                self.net.save_parameters(self.param_save_path + 'best_{:s}_{:s}.params'.format(
                                            'IoU', self.save_prefix))
                # log the epoch number and the best IoU value
                with open(self.param_save_path + 'best_IoU.log', 'a') as f:
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # time for log message
                    f.write('{} - epoch: {:04d} IoU: {:.4f}\n'.format(dt_string, epoch, self.best_iou))

                print("Best IoU {:.4f} model saved".format(self.best_iou))

            if nIoU > self.best_nIoU:
                self.best_nIoU = nIoU
                # save the best model
                self.net.save_parameters(self.param_save_path + 'best_{:s}_{:s}.params'.format(
                                            'nIoU', self.save_prefix))
                # log the epoch number and the best IoU value
                with open(self.param_save_path + 'best_nIoU.log', 'a') as f:
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    f.write('{} - epoch: {:04d} nIoU: {:.4f}\n'.format(dt_string, epoch, self.best_nIoU))
                print("Best nIoU {:.4f} model saved".format(self.best_nIoU))

            if epoch >= args.epochs - 1:
                print("best_iou: {:.4f}".format(self.best_iou))
                print("best_nIoU: {:.4f}".format(self.best_nIoU))

                # with open(self.save_prefix + '_' + '_GPU_' + self.args.gpus +
                #           '_best_IoU.log', 'a') as f:
                #     f.write('Finished\n')
                # self.net.save_parameters('tmp_{:s}_best_{:s}_{:s}.params'.format(
                #     self.save_prefix, 'IoU', str(self.best_iou)))
                # self.net.save_parameters('tmp_{:s}_best_{:s}_{:s}.params'.format(
                #     self.save_prefix, 'nIoU', str(self.best_nIoU)))
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                with open(self.param_save_path + self.date_string + self.save_prefix + '_best_IoU.log', 'a') as f:
                    f.write('{} - Finished {:04d} epoch, best IoU: {:.4f}\n'.format(dt_string, epoch, self.best_iou))
                with open(self.param_save_path + self.date_string + self.save_prefix + '_best_nIoU.log', 'a') as f:
                    f.write('{} - Finished {:04d} epoch, best nIoU: {:.4f}\n'.format(dt_string, epoch, self.best_nIoU))
                
                # Save plot for loss and iou
                if epoch % 5 == 0:
                    # for loss
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(range(epoch+1), self.train_losses)
                    ax.plot(range(epoch+1), self.val_losses)
                    ax.plot(range(epoch+1), self.ious)
                    ax.plot(range(epoch+1), self.nious)
                    ax.legend(["Train loss","Val loss","IoU","nIoU"])
                    ax.set_xlabel("Epoch")
                    fig.savefig(self.param_save_path + "IoU_losses.png")
                    plt.close('all')
        # evaluation
        else:
            # evaluation log
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open(self.param_save_path + 'evaluation.log', 'a') as f:
                f.write('{} - {} result - loss: {:.4f} IoU: {:.4f}, nIoU: {:.4f}\n'.format(dt_string, args.val_split, val_loss_ave, IoU, nIoU))


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
