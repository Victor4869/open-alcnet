import os
# os.system("taskset -p -c 1-96 %d" % os.getpid())
import scipy.misc
import platform
import timeit
import sys
import socket
import argparse
import numpy as np
# from utils import summary
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

from data import IceContrast
from model import MPCMResNetFPN
from loss import SoftIoULoss

from datetime import datetime
import time

import matplotlib.pyplot as plt
import cv2

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')

    # host information
    parser.add_argument('--host', type=str, default='xxx',
                        help='xxx is a place holder, leave as default or replace with \
                            your host information. e.g. host name or GPU details')

    # model
    parser.add_argument('--net-choice', type=str, default='MPCMResNetFPN',
                        help='model name PCMNet, PlainNet')
    parser.add_argument('--pyramid-mode', type=str, default='Dec',
                        help='Inc, Dec')
    parser.add_argument('--scale-mode', type=str, default='Multiple',
                        help='Single, Multiple, Selective')
    parser.add_argument('--pyramid-fuse', type=str, default='bottomuplocal',
                        help='add, max, sk')
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
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1,
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
    parser.add_argument('--metric', type=str, default='mAP',
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
        args.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        print('Number of GPUs:', len(args.ctx))

    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
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
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        ################################# dataset and dataloader #################################
        
        # if platform.system() == "Darwin":
        #     data_root = os.path.join('~', 'Nutstore Files', 'Dataset')
        # elif platform.system() == "Linux":
        #     data_root = os.path.join('~', 'datasets')
        #     if args.colab:
        #         # data_root = '/content/datasets'
        #         data_root = '/content/drive/MyDrive/Colab Notebooks/alcnet/datasets'
        # else:
        #     raise ValueError('Notice Dataset Path')
        
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

        # data_kwargs = {'base_size': args.base_size,
        #                'crop_size': args.crop_size, 'root': data_root,
        #                'base_dir' : args.dataset}

        valset = IceContrast(split=args.val_split, mode='testval', include_name=True,
                             **data_kwargs)
        self.valset = valset

        net_choice = args.net_choice
        print("net_choice: ", net_choice)

        model = '' #Fix scope issue

        if net_choice == 'MPCMResNetFPN':
            layers = [self.args.blocks] * 3
            channels = [8, 16, 32, 64]
            shift = self.args.shift
            pyramid_mode = self.args.pyramid_mode
            scale_mode = self.args.scale_mode
            pyramid_fuse = self.args.pyramid_fuse

            model = MPCMResNetFPN(layers=layers, channels=channels, shift=shift,
                                  pyramid_mode=pyramid_mode, scale_mode=scale_mode,
                                  pyramid_fuse=pyramid_fuse, classes=valset.NUM_CLASS)
            print("net_choice: ", net_choice)
            print("scale_mode: ", scale_mode)
            print("pyramid_fuse: ", pyramid_fuse)
            print("layers: ", layers)
            print("channels: ", channels)
            print("shift: ", shift)
        else:
            raise ValueError('Unknow net_choice')

        if args.host == 'xxx':
            self.host_name = socket.gethostname()  # automatic
        else:
            self.host_name = args.host # Specified host information
            


        # self.save_prefix = 'MLCPFN' + '_' + args.scale_mode + '_' + args.pyramid_fuse + '_'
        self.save_prefix = net_choice + '_' + args.scale_mode + '_' + args.pyramid_fuse + '_'

        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            # model.initialize(init=init.Xavier(), ctx=args.ctx, force_reinit=True)
            model.initialize(init=init.MSRAPrelu(), ctx=args.ctx, force_reinit=True)
            print("Model Initializing")
            print("args.ctx: ", args.ctx)

        # params_path = '/content/drive/MyDrive/Colab Notebooks/alcnet/params/' + self.paramsfile
        # model.load_parameters(params_path, ctx=args.ctx)
        self.net = model

        # create criterion
        kv = mx.kv.create(args.kvstore)

        optimizer_params = {
            'wd': args.weight_decay,
            'learning_rate': args.lr
        }

        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

    ################################# evaluation metrics #################################

    def validation(self, epoch):
        
        # setting save path for prediction images
        # using the parameter file name as the folder name for predicted images
        head, tail = os.path.split(args.resume)
        f_name, f_ext = os.path.splitext(tail)

        path = os.path.join(self.alc_dir,'results', f_name + "/")
        save_path = os.path.expanduser(path)
        try:
            os.makedirs(save_path)
            print(save_path + ' did not existed and was created.')
        except:
            print("Image folder already found: " + save_path)

        times = [] # to record inference time

        
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0' # turn off performance tests 
        for img, mask, img_id in self.valset:

            exp_img = img.expand_dims(axis=0) # load image

            #exp_img to gpu if there is one
            if not (len(mx.test_utils.list_gpus()) == 0):
                exp_img = exp_img.copyto(mx.gpu())
            tic = time.time()
            pred = self.net(exp_img)  # prediction
            mx.nd.waitall()
            toc = time.time()
            times.append(toc-tic) # record the inference time
            
            pred = pred.squeeze().asnumpy() > 0 # convert NDarray to boolean numpy array

            #save image
            plt.imsave(save_path + img_id + '.png', pred)

            # for saving balck and white image
            # plt.imsave(save_path + img_id + '.png', pred, cmap='gray', vmin=0, vmax=1)

            # draw bounding box using the ground true labelling xml file
            # pass if xml file is not found
            try:
              xml_path = ''
              if self.dataset == 'sirst':
                  xml_path = os.path.join(self.data_root,self.dataset,'masks', img_id+'.xml')
                  xml_path = os.path.expanduser(xml_path)
              # print(xml_path)
              xml_file = ET.parse(xml_path)
              xml_root = xml_file.getroot()
              # sample_annotations = []
              width = int(xml_root.find('size').find('width').text)
              height = int(xml_root.find('size').find('height').text)

              for neighbor in xml_root.iter('bndbox'):
                  xmin = int(int(neighbor.find('xmin').text)/width*512)
                  ymin = int(int(neighbor.find('ymin').text)/height*512)
                  xmax = int(int(neighbor.find('xmax').text)/width*512)
                  ymax = int(int(neighbor.find('ymax').text)/height*512)
                  
                  # sample_annotations.append([xmin, ymin, xmax, ymax])
                  img = cv2.imread(save_path + img_id + '.png') # load the image saved above
                  box_offset = 1 # offset the bounding box from the centre by the specified pixels

                  # draw bounding box to image
                  image2 = cv2.rectangle(img,(xmin-box_offset, ymin-box_offset), 
                                  (xmax+box_offset, ymax+box_offset), (0,0,255),1)
                  cv2.imwrite(save_path + 'boxed_'+ img_id + '.png', image2) # save the new image
            except:
              pass
            # break

        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1' # turn performance test back on

        # create log file
        str1 = "Dateset: " + args.dataset + ", Prediction time: {:.4f}s, FPS: {:.2f}".format(sum(times), len(self.valset)/sum(times) )
        str2 = "Log file and " + str(len(self.valset)) + " images saved in: " + save_path
        now = datetime.now()
        with open(save_path + 'visual.log', 'a') as f:
          f.write("\n" "Date: "+ now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
          f.write("Parameter file: " + args.resume + "\n")
          f.write("Host info: " + self.host_name + "\n")
          f.write(str1 + "\n")
          f.write(str2 + "\n")

        # print summary on screen
        print(str1)
        print(str2)

        # save_path = os.path.expanduser('/Users/grok/Downloads/')

        # # summary(self.net, mx.nd.zeros((1, 3, args.crop_size, args.crop_size), ctx=args.ctx[0]))
        # # sys.exit()

        # i = 0
        # mx.nd.waitall()
        # start = timeit.default_timer()
        # for img, mask, img_id in self.valset:
        #     exp_img = img.expand_dims(axis=0)
        #     if not (len(mx.test_utils.list_gpus()) == 0):
        #         exp_img = exp_img.copyto(mx.gpu())
        #     # pred = self.net(exp_img).squeeze().asnumpy() > 0
        #     pred = self.net(exp_img)
        #     pred = pred.squeeze().asnumpy() > 0 # from NDArray to a boolean numpy array
        #     plt.imsave(save_path + img_id + '.png', pred)
        #     # print(pred.shape)

        # # save_path = os.path.expanduser('/Users/grok/Downloads/img')
        # # for img, mask, img_id in self.valset:
        #     # exp_img = img.expand_dims(axis=0)
        #     # img = mx.nd.transpose(img, (1, 2, 0))
        #     # print(img.shape)
        #     # img = img.squeeze().asnumpy() / 255
        #     # plt.imsave(save_path + img_id + '.png', img)


        #     # break

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        trainer.validation(0)
