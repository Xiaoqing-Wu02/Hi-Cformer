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

    model = hicformer(whole_size=whole_size, img_size=img_size, patch_size=patch_size, encoder_dim=encoder_dim,hidden_dim=hidden_dim,  chr_num=len(chr_list),in_chans=1,
                 embed_dim=embed_dim, depth=depth, num_heads=8,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), chr_mutual_visibility=vis, weight=weight, chr_single_loss=chr_single_loss,num_patches=num_patches, otherlf=otherlf, enorm=enorm)
    device = "cuda:"+str(cuda)
    model.to(device)

    labels = new_label
    cell_num = len(labels)
    cell_feats1 = torch.zeros((cell_num, 1), dtype=torch.float32)


    for i in range(cell_num):
        total_non_zero_sum = 0
        total_non_zero_count = 0


        for chrom_matrix in all_pics:
            cell_matrix = chrom_matrix[i, :, :]  
            non_zero_entries = cell_matrix[cell_matrix > 0]  


            total_non_zero_sum += torch.sum(non_zero_entries).item()  

            total_non_zero_count += non_zero_entries.numel() 

        if total_non_zero_count > 0:
            cell_feats1[i, 0] = total_non_zero_sum / total_non_zero_count
        else:
            cell_feats1[i, 0] = 0  
    all_pics.append(torch_tensor)
    labels = new_label
    all_pics.append(cell_feats1)
    dataset = TensorDataset(*all_pics)
    data_loader_test = DataLoader(dataset, batch_size=16, shuffle=False)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)




    

    all_states = {}
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    data_loader_train = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)


    print(optimizer)
    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    best_loss =  float("inf")
    p_count = 0

    if pretrain == True:
        optimizer_chr = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        optimizer_patch = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

        
        
        for epoch in range(pretrain_epoch):
            if (epoch+5) % 10 == 0:
                nmi, ari,homo,ami = calNMI_pretrain(model, data_loader_test, labels, device=device)
                mnmi = max(nmi)
                if mnmi > best_nmi:
                    best_nmi = mnmi
                    save_name = f"{save_name_prefix}{num}dim_{hidden_dim}.pth"
                    if os.path.exists(save_name):
                        all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
                    else:
                        all_states = {}
                    all_states['best_cluster_pretrain'] = {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer_state_dict': optimizer_chr.state_dict(),
                            }
                    torch.save(all_states,  save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
                with open(save_name_prefix+"result"+str(num)+".txt", 'a') as f:
                    f.write(str(epoch) + "chr_pretrain"+ '\n')
                    f.write(f"NMI: {nmi[0]} ARI: {ari[0]} Homo: {homo[0]} AMI: {ami[0]}\n")
                    f.write(f"NMI: {nmi[1]} ARI: {ari[1]} Homo: {homo[1]} AMI: {ami[1]}\n")
            model.train()
            epoch_chr_loss = 0.0
            epoch_chr_single_loss = 0.0
            epoch_loss = 0.0
            for inputs in data_loader_train:
                inputs = [item.to(device) for item in inputs]
                optimizer_chr.zero_grad()
                chr_loss, chr_single_loss = model.aepretrain(inputs)
                loss = 0.0*chr_single_loss + 1.0*chr_loss
                loss.backward()
                optimizer_chr.step()
                epoch_chr_loss += chr_loss.item()
                if model.chr_single_loss:
                    epoch_chr_single_loss += chr_single_loss.item()
                else:
                    epoch_chr_single_loss += chr_single_loss
                epoch_loss += loss.item()
                #noam_scheduler.step()
            l = len(data_loader_train)
            with open(save_name_prefix+"loss"+str(num)+".txt", 'a') as f:
                f.write("Epoch {}/{} - Chr Loss: {:.4f} - Chr single Loss: {:.4f} - Best Loss: {:.4f}".format(epoch+1, pretrain_epoch, epoch_chr_loss/l, epoch_chr_single_loss/l, best_loss) + '\n')
            print("Epoch {}/{} - Chr Loss: {:.4f} - Chr single Loss: {:.4f} - Best Loss: {:.4f}".format(epoch+1, pretrain_epoch, epoch_chr_loss/l, epoch_chr_single_loss/l, best_loss) + '\n')
            
            if epoch_loss / len(data_loader_train) < best_loss:
                best_loss = epoch_loss / len(data_loader_train)
                p_count = 0
                save_name = f"{save_name_prefix}{num}dim_{hidden_dim}.pth"
                if os.path.exists(save_name):
                    all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
                else:
                    all_states = {}
                all_states['best_pretrain'] = {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer_state_dict': optimizer_chr.state_dict(),
                            }
                torch.save(all_states,  save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
            else:
                p_count = p_count + 1
            epoch = epoch + 1
            if p_count > patience:
                p_count = 0
                break  
        
        best_loss =  float("inf")
        p_count = 0
        for epoch in range(pretrain_epoch):
            model.train()
            epoch_chr_loss = 0.0
            for inputs in data_loader_train:
                inputs = [item.to(device) for item in inputs]
                optimizer_patch.zero_grad()
                chr_loss = model.patch_pretrain(inputs)
                #loss = loss_ratio*patch_loss + (1-loss_ratio)*chr_loss
                chr_loss.backward()
                optimizer_patch.step()
                epoch_chr_loss += chr_loss.item()
            l = len(data_loader_train)
            with open(save_name_prefix+"loss"+str(num)+".txt", 'a') as f:
                f.write("Epoch {}/{} - Patch Loss: {:.4f} - Best Loss: {:.4f}".format(epoch+1, pretrain_epoch, epoch_chr_loss/l, best_loss) + '\n')
            print("Epoch {}/{} - Patch Loss: {:.4f} - Best Loss: {:.4f}".format(epoch+1, pretrain_epoch, epoch_chr_loss/l, best_loss))
            
            if epoch_chr_loss / len(data_loader_train) < best_loss:
                save_name = f"{save_name_prefix}{num}dim_{hidden_dim}.pth"
                if os.path.exists(save_name):
                    all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
                else:
                    all_states = {}
                all_states['best_pretrain'] = {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer_state_dict': optimizer_chr.state_dict(),
                            }
                torch.save(all_states,  save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
                best_loss = epoch_chr_loss / len(data_loader_train)
                p_count = 0
            else:
                p_count = p_count + 1
            if p_count > patience:
                p_count = 0
                break    
                #noam_scheduler.step()
        all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
        model.load_state_dict(all_states['best_pretrain']['model'])
        model.to(device)
    elif pretrain != False:
        all_states = torch.load(save_name_prefix+pretrain+"dim_"+str(hidden_dim)+".pth")
        model.load_state_dict(all_states['best_pretrain']['model'])
        nmi, ari,homo,ami = calNMI_pretrain(model, data_loader_test, labels, device=device)
        model.to(device)
    else:
        model.to(device)
        
            
    best_loss =  float("inf")
    p_count = 0
        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    data_loader_test = DataLoader(dataset, batch_size=16, shuffle=False)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=atlr, weight_decay=weight_decay, betas=betas)
    


    cudnn.benchmark = True


    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    data_loader_train = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)


    print(optimizer)
    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    best_loss =  float("inf")
    best_loss =  float("inf")
    p_count = 0
    p_count_all = 0
    best_loss_chr = float("inf")
    data_loader_train = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    for epoch in range(epochs):
        if (epoch) % 10 == 0:
            nmi, ari,homo,ami = calNMI(model, data_loader_test, labels, device=device)
            mnmi = nmi[0]
            with open(save_name_prefix+"result"+str(num)+".txt", 'a') as f:
                f.write(str(epoch) + '\n')
                f.write(f"NMI: {nmi[0]} ARI: {ari[0]} Homo: {homo[0]} AMI: {ami[0]}\n")
            if mnmi > best_nmi:
                save_name = f"{save_name_prefix}{num}dim_{hidden_dim}.pth"
                if os.path.exists(save_name):
                    all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
                else:
                    all_states = {}
                all_states['best_cluster'] = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
                best_nmi = mnmi
                torch.save(all_states,  save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
        model.train()
        running_loss = 0.0
        epoch_patch_loss = 0.0
        epoch_chr_loss = 0.0
        epoch_chr_single_loss = 0.0
        for inputs in data_loader_train:
            inputs = [item.to(device) for item in inputs]
            optimizer.zero_grad()
            x, chr_pred, cell_embed,token_class, patch_loss, chr_loss, chr_single_loss = model(inputs, mask_ratio=mask_ratio)
            loss = loss_ratio*patch_loss + (1-loss_ratio)*(1-chr_single_ratio)*chr_loss + (1-loss_ratio)*(chr_single_ratio)*chr_single_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_patch_loss += patch_loss.item()
            epoch_chr_loss += chr_loss.item()
            if chr_single_loss:
                epoch_chr_single_loss += chr_single_loss.item()
            else:
                epoch_chr_single_loss += chr_single_loss
        l = len(data_loader_train)
        with open(save_name_prefix+"loss"+str(num)+".txt", 'a') as f:
            f.write("Epoch {}/{} - Loss: {:.4f} - Patch Loss: {:.4f} - Chr Loss: {:.4f} - Chr single Loss: {:.4f} - Best Loss: {:.4f}".format(epoch+1, epochs, running_loss/l, epoch_patch_loss/l, epoch_chr_loss/l, epoch_chr_single_loss/l, best_loss) + '\n')
        print("Epoch {}/{} - Loss: {:.4f} - Patch Loss: {:.4f} - Chr Loss: {:.4f} - Chr single Loss: {:.4f} - Best Loss: {:.4f}".format(epoch+1, epochs, running_loss/l, epoch_patch_loss/l, epoch_chr_loss/l, epoch_chr_single_loss/l, best_loss))
        
        if running_loss / len(data_loader_train) < best_loss:
            best_loss = running_loss / len(data_loader_train)
            p_count = 0
            save_name = f"{save_name_prefix}{num}dim_{hidden_dim}.pth"
            if os.path.exists(save_name):
                all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
            else:
                all_states = {}
            all_states['best'] = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(all_states,  save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
        
        else:
            p_count = p_count + 1
        if epoch_chr_loss / len(data_loader_train) < best_loss_chr:
            best_loss_chr = epoch_chr_loss / len(data_loader_train)
            p_count_all = 0
            save_name = f"{save_name_prefix}{num}dim_{hidden_dim}.pth"
            if os.path.exists(save_name):
                all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
            else:
                all_states = {}
            all_states['best_chr'] = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(all_states,  save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
        else:
            p_count_all = p_count_all + 1
        epoch = epoch + 1
        if p_count > patience and p_count_all > patience:
            p_count = 0
            break   
    if os.path.exists(save_name):
        all_states = torch.load(save_name_prefix+str(num)+"dim_"+str(hidden_dim)+".pth")
        model.load_state_dict(all_states['best_chr']['model'])
        nmi, ari,homo,ami = calNMI(model, data_loader_test, labels, device=device)
        with open(save_name_prefix+"result"+str(num)+".txt", 'a') as f:
            f.write('best model' + '\n')
            f.write(f"NMI: {nmi[0]} ARI: {ari[0]} Homo: {homo[0]} AMI: {ami[0]}\n")
            f.write(f"NMI: {nmi[1]} ARI: {ari[1]} Homo: {homo[1]} AMI: {ami[1]}\n")





