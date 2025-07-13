label_path = "/home/wuxiaoqing/scHi-C_dataset/Ramani/label_info.pickle"
label_name = "cell type"
raw_path = "/home/wuxiaoqing/scHi-C_dataset/Ramani/Temp/raw"
save_path = "/home/wuxiaoqing/Hi-Cformer/demo"
save_name_prefix = save_path + "/hicformer_"
chr_list = ['chr1', 'chr2',  'chr3', 'chr4','chr5', 'chr6','chr7','chr10','chr8','chr14',
            'chr9', 'chr11', 'chr13', 'chr12', 'chr15', 'chr16','chr17',  'chr18', 'chr19', 'chr20', 'chr21', 'chr22','chrX']

import argparse
import datetime
import json
import os
import pickle
import time
from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import DataLoader, TensorDataset

from .utils import cal_expand_tensor, calNMI_pretrain, calNMI
from .model import hicformer

if __name__ == '__main__':
    with open(label_path, 'rb') as file:
        datas = pickle.load(file)
    labels = datas[label_name]
    folder_path = raw_path
    all_pics = []
    new_label = [labels[i] for i in range(len(labels))]

    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999])
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--pretrain_epoch', type=int, default=150)
    parser.add_argument('--chr_single_ratio', type=float, default=0.5)
    parser.add_argument('--loss_ratio', type=float, default=0.02)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num', type=int, default=11)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--pretrain', type=str, default='True')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--vis', type=str, default='True')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--best_nmi', type=int, default=0)
    parser.add_argument('--chr_single_loss',  type=str, default='True')
    parser.add_argument('--encoder_dim', nargs='+', type=int, default=[])
    parser.add_argument('--weight', type=float, default=0.1)
    parser.add_argument('--patch_size', nargs='+', type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--atlr', type=float, default=5e-4)
    parser.add_argument('--otherlf', type=str, default='MSE')
    parser.add_argument('--enorm',  type=str, default='False')
    args = parser.parse_args()
    otherlf = args.otherlf
    betas = args.betas
    img_size = args.img_size
    patience = args.patience
    lr = args.lr
    cuda = args.cuda
    pretrain_epoch = args.pretrain_epoch
    chr_single_ratio = args.chr_single_ratio
    loss_ratio = args.loss_ratio
    embed_dim = args.embed_dim
    num = args.num
    depth = args.depth
    enorm = True if args.enorm.lower() == 'true' else False
    if args.pretrain.lower() == 'true':
        pretrain = True
    elif args.pretrain.lower() == 'false':
        pretrain = False
    else:
        pretrain = args.pretrain
    hidden_dim = args.hidden_dim
    vis =  True if args.vis.lower() == 'true' else False
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    best_nmi = args.best_nmi
    chr_single_loss = True if args.chr_single_loss.lower() == 'true' else False
    encoder_dim = args.encoder_dim
    weight = args.weight
    patch_size = args.patch_size
    train_ratio = args.train_ratio
    mask_ratio = args.mask_ratio
    atlr = args.atlr
    seed = 1
    best_loss =  float("inf")
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    print("pretrain:",pretrain)
    print("vis:", vis)
    params = {
        "pretrain_epoch": pretrain_epoch,
        "loss_ratio": loss_ratio,
        "embed_dim": embed_dim,
        "num": num,
        "depth": depth,
        "pretrain": pretrain,
        "hidden_dim": hidden_dim,
        "vis": vis,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "betas": betas,
        "patience":patience,
        "learning rate":lr,
        "chr_single_loss":chr_single_loss,
        "encoder_dim":encoder_dim,
        "weight":weight,
        "patch_size":patch_size,
        "chr_single_ratio":chr_single_ratio,
        "mask_ratio":mask_ratio,
        "atlr":atlr,
        "otherlf":otherlf,
        "enorm":enorm
    }

    filename = save_name_prefix+f"params_{params['num']}.json"
    with open(filename, 'w') as f:
        json.dump(params, f)
    device = "cuda:"+str(cuda)

    chr_cls = []
    all_pics = []
    folder_path = raw_path
    new_label = [labels[i] for i in range(len(labels)) ]
    max_height = 0
    for chrname in chr_list:
        
        f = folder_path + '/' + chrname + '_sparse_adj.npy'
        loaded_data = np.load(f, allow_pickle=True)
        onechr_pic = [loaded_data[i].toarray() for i in range(len(labels)) ]
        datas = np.array(onechr_pic)
        if datas.shape[1] > max_height:
            max_height = datas.shape[1]
        datas = datas.reshape(datas.shape[0], -1)
        onechr_pic = torch.stack([torch.tensor(pic, dtype=torch.float32).unsqueeze(0) for pic in onechr_pic])
        all_pics.append(onechr_pic)
        pca = PCA(n_components=embed_dim,random_state=3)
        result = pca.fit_transform(datas)
        scaler = StandardScaler()
        if enorm:
            result = scaler.fit_transform(result)
        chr_cls.append(result)
        print(chrname)
    list_of_tensors = [torch.from_numpy(arr) for arr in chr_cls]
    torch_tensor = torch.stack(list_of_tensors, dim=1)
    whole_size = 0
    for item in all_pics:
        whole_size = whole_size + cal_expand_tensor(item)
    num_patches = max_height // min(patch_size)

    model1 = hicformer(whole_size=whole_size, img_size=img_size, patch_size=patch_size, encoder_dim=encoder_dim,hidden_dim=hidden_dim,  chr_num=len(chr_list),in_chans=1,
                 embed_dim=embed_dim, depth=depth, num_heads=8,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), chr_mutual_visibility=vis, weight=weight, chr_single_loss=chr_single_loss,num_patches=num_patches, otherlf=otherlf, enorm=enorm)
    device = "cuda:"+str(cuda)
    model1.to(device)
    print(model1)