from __future__ import print_function
import os, random, time, math
import numpy as np
import torch
import timm
from timm.data import resolve_data_config





class value():
    def __init__(self):
        self.dataset = 'mvtec' #dataset name: mvtec/stc (default: mvtec)'
    
        self.checkpoint = 'C:/Users/yusuk/cflow-ad3/weights/mvtec_wide_resnet50_2_freia_cflow_pl3_cb8_inp512_run0_leather_2022_03_18_18_45_56_785.pt' #file with saved checkpoint
        self.coupling_blocks = 8 #number of layers used in NF model (default: 8)
        self.run_name = 0 #name of the run (default: 0)
        self.input_size = 512 #image resize dimensions (default: 256)
        self.action_type = 'norm-test' #norm-train/norm-test (default: norm-train)
        self.gpu = '0' #GPU device number
        self.no_cuda = False #disables CUDA training
        self.video_path ='.' #video file path
        self.img_dims = 0
        self.clamp_alpha = 1.9
        self.dropout = 0.0
        self.print_freq = 2
        self.temp =0.5
        self.lr_decay_epochs = 0
        self.lr_decay_rate = 0.1
        self.lr_warm_epochs = 2
        self.lr_warm = True
        self.lr_cosine = True
        self.lr_warmup_from = 0
        self.lr_warmup_to = 0

        self.data_path = './data/MVTec-AD/leather/train/good'
        self.crop_size = 0
        self.img_size = 0
        self.norm_mean = 0
        self.norm_std = 0
        self.condition_vec = 128
        self.sub_epochs = 8 #number of sub epochs to train (default: 8)
        self.enc_arch = 'wide_resnet50_2' #feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: wide_resnet50_2)
        self.dec_arch = 'freia-cflow' #normalizing flow model (default: freia-cflow)
        self.pool_layers = 3 #number of layers used in NF model (default: 3)
        self.lr = 2e-4 #learning rate (default: 2e-4)
        self.workers = 4 #number of data loading workers (default: 4)
        self.batch_size = 32 #train batch size (default: 32)
        self.meta_epochs = 25 #number of meta epochs to train (default: 25)
        self.msg = ""




def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





def set_value():

    c = value()

    ###########image
    c.img_size = (c.input_size, c.input_size)  # HxW format
    c.crp_size = (c.input_size, c.input_size)  # HxW format
    c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    c.img_dims = [3] + list(c.img_size)
    ###########

    ###########network hyperparameters
    c.clamp_alpha = 1.9  # see paper equation 2 for explanation
    c.condition_vec = 128
    c.dropout = 0.0  # dropout in s-t-networks
    ###########

    ###########unsup-train
    c.print_freq = 2
    c.temp = 0.5
    c.lr_decay_epochs = [i*c.meta_epochs//100 for i in [50,75,90]]
    c.msg = 'LR schedule: {}'.format(c.lr_decay_epochs)
    c.lr_decay_rate = 0.1
    c.lr_warm_epochs = 2
    c.lr_warm = True
    c.lr_cosine = True
    if c.lr_warm:
        c.lr_warmup_from = c.lr/10.0
        if c.lr_cosine:
            eta_min = c.lr * (c.lr_decay_rate ** 3)
            c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                    1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)) / 2
        else:
            c.lr_warmup_to = c.lr
    ########

    ###########GPU
    #c.use_cuda
    #c.device
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    c.device = torch.device("cuda" if c.use_cuda else "cpu")
    ###########

    return c
