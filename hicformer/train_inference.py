import argparse
import json
import os
import pickle
from functools import partial

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import cal_expand_tensor
from .model import hicformer


# --------------------------
# Unified config defaults
# --------------------------
DEFAULT_CONFIG = {
    "label_path": "/home/wuxiaoqing/scHi-C_dataset/Ramani/label_info.pickle",
    "label_name": "cell type",
    "raw_path": "/home/wuxiaoqing/scHi-C_dataset/Ramani/Temp/raw",
    "save_path": "/home/wuxiaoqing/revision/time",
    "save_name_prefix": None,  # if None, will be save_path + "/hicformer"
    "chr_list": [
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr10", "chr8",
        "chr14", "chr9", "chr11", "chr13", "chr12", "chr15", "chr16", "chr17",
        "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"
    ],

    "betas": [0.9, 0.999],
    "img_size": 112,
    "patience": 10,
    "lr": 5e-4,
    "cuda": 1,
    "pretrain_epoch": 150,
    "chr_single_ratio": 0.5,
    "loss_ratio": 0.02,
    "embed_dim": 128,
    "num": 11,
    "depth": 4,
    "pretrain": "True",
    "hidden_dim": 64,
    "vis": "True",
    "batch_size": 32,
    "weight_decay": 0.0,
    "epochs": 180,
    "chr_single_loss": "True",
    "encoder_dim": [],
    "weight": 0.1,
    "patch_size": [8, 16, 32, 64, 128],
    "train_ratio": 0.9,
    "mask_ratio": 0.5,
    "atlr": 5e-4,
    "otherlf": "MSE",
    "enorm": "False",
}


def load_config(config_path):
    cfg = DEFAULT_CONFIG.copy()
    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)

    if cfg["save_name_prefix"] is None:
        cfg["save_name_prefix"] = os.path.join(cfg["save_path"], "hicformer")

    return cfg


def build_parser(default_cfg):
    parser = argparse.ArgumentParser(
        description='Train and inference for HiCFormer.'
    )

    # config file
    parser.add_argument('--config', type=str, default=None)

    # keep original parameter names, but do not hardcode defaults here anymore
    parser.add_argument('--label_path', type=str, default=None)
    parser.add_argument('--label_name', type=str, default=None)
    parser.add_argument('--raw_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--save_name_prefix', type=str, default=None)
    parser.add_argument('--chr_list', nargs='+', type=str, default=None)

    parser.add_argument('--betas', nargs='+', type=float, default=None)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--cuda', type=int, default=None)
    parser.add_argument('--pretrain_epoch', type=int, default=None)
    parser.add_argument('--chr_single_ratio', type=float, default=None)
    parser.add_argument('--loss_ratio', type=float, default=None)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--vis', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--chr_single_loss', type=str, default=None)
    parser.add_argument('--encoder_dim', nargs='+', type=int, default=None)
    parser.add_argument('--weight', type=float, default=None)
    parser.add_argument('--patch_size', nargs='+', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--atlr', type=float, default=None)
    parser.add_argument('--otherlf', type=str, default=None)
    parser.add_argument('--enorm', type=str, default=None)

    return parser


def merge_config_and_args(cfg, args):
    final_cfg = cfg.copy()
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            final_cfg[k] = v

    if final_cfg["save_name_prefix"] is None:
        final_cfg["save_name_prefix"] = os.path.join(final_cfg["save_path"], "Ramani")

    return final_cfg


if __name__ == '__main__':
    # --------------------------
    # Load config first
    # --------------------------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    cfg = load_config(pre_args.config)
    parser = build_parser(cfg)
    args = parser.parse_args()
    cfg = merge_config_and_args(cfg, args)

    # keep original variable names
    label_path = cfg["label_path"]
    label_name = cfg["label_name"]
    raw_path = cfg["raw_path"]
    save_path = cfg["save_path"]
    save_name_prefix = cfg["save_name_prefix"]
    chr_list = cfg["chr_list"]

    # --------------------------
    # Original loading
    # --------------------------
    with open(label_path, 'rb') as file:
        datas = pickle.load(file)
    labels = datas[label_name]
    folder_path = raw_path
    new_label = [labels[i] for i in range(len(labels))]

    otherlf = cfg["otherlf"]
    betas = cfg["betas"]
    img_size = cfg["img_size"]
    patience = cfg["patience"]
    lr = cfg["lr"]
    cuda = cfg["cuda"]
    pretrain_epoch = cfg["pretrain_epoch"]
    chr_single_ratio = cfg["chr_single_ratio"]
    loss_ratio = cfg["loss_ratio"]
    embed_dim = cfg["embed_dim"]
    num = cfg["num"]
    depth = cfg["depth"]
    enorm = True if str(cfg["enorm"]).lower() == 'true' else False

    if str(cfg["pretrain"]).lower() == 'true':
        pretrain = True
    elif str(cfg["pretrain"]).lower() == 'false':
        pretrain = False
    else:
        pretrain = cfg["pretrain"]  # load pretrain weights by string (kept for compatibility)

    hidden_dim = cfg["hidden_dim"]
    vis = True if str(cfg["vis"]).lower() == 'true' else False
    batch_size = cfg["batch_size"]
    weight_decay = cfg["weight_decay"]
    epochs = cfg["epochs"]
    chr_single_loss_flag = True if str(cfg["chr_single_loss"]).lower() == 'true' else False
    encoder_dim = cfg["encoder_dim"]
    weight = cfg["weight"]
    patch_size = cfg["patch_size"]
    train_ratio = cfg["train_ratio"]
    mask_ratio = cfg["mask_ratio"]
    atlr = cfg["atlr"]

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False

    # device
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # --------------------------
    # Build data tensors
    # --------------------------
    chr_cls = []
    all_pics = []
    max_height = 0
    pca_save_dir = os.path.join(folder_path, "pca_results")
    os.makedirs(pca_save_dir, exist_ok=True)

    for chrname in chr_list:
        f = folder_path + '/' + chrname + '_sparse_adj.npy'
        loaded_data = np.load(f, allow_pickle=True)
        onechr_pic = [loaded_data[i].toarray() for i in range(len(labels))]
        datas_np = np.array(onechr_pic)

        if datas_np.shape[1] > max_height:
            max_height = datas_np.shape[1]

        datas_flat = datas_np.reshape(datas_np.shape[0], -1)

        onechr_pic_t = torch.stack([torch.tensor(pic, dtype=torch.float32).unsqueeze(0) for pic in onechr_pic])
        all_pics.append(onechr_pic_t)

        pca_file = os.path.join(pca_save_dir, f"{chrname}_sparsePCA.npy")
        if os.path.exists(pca_file):
            result = np.load(pca_file)
        else:
            pca = TruncatedSVD(n_components=embed_dim, random_state=3)
            result = pca.fit_transform(datas_flat)
            result = result.astype(np.float32)
            np.save(pca_file, result)

        if enorm:
            scaler = StandardScaler()
            result = scaler.fit_transform(result)

        chr_cls.append(result)
        print(chrname)

    list_of_tensors = [torch.from_numpy(arr) for arr in chr_cls]
    torch_tensor = torch.stack(list_of_tensors, dim=1)

    whole_size = 0
    for item in all_pics:
        whole_size = whole_size + cal_expand_tensor(item)

    num_patches = max_height // min(patch_size)

    model = hicformer(
        whole_size=whole_size, img_size=img_size, patch_size=patch_size,
        encoder_dim=encoder_dim, hidden_dim=hidden_dim,
        chr_num=len(chr_list), in_chans=1,
        embed_dim=embed_dim, depth=depth, num_heads=8,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        chr_mutual_visibility=vis, weight=weight,
        chr_single_loss=chr_single_loss_flag, num_patches=num_patches,
        otherlf=otherlf, enorm=enorm
    )
    model.to(device)

    # cell_feats1 (as original)
    labels = new_label
    cell_num = len(labels)
    cell_feats1 = torch.zeros((cell_num, 1), dtype=torch.float32)

    for i in range(cell_num):
        total_non_zero_sum = 0.0
        total_non_zero_count = 0
        for chrom_matrix in all_pics:
            cell_matrix = chrom_matrix[i, :, :]
            non_zero_entries = cell_matrix[cell_matrix > 0]
            total_non_zero_sum += torch.sum(non_zero_entries).item()
            total_non_zero_count += non_zero_entries.numel()
        if total_non_zero_count > 0:
            cell_feats1[i, 0] = total_non_zero_sum / total_non_zero_count
        else:
            cell_feats1[i, 0] = 0.0

    all_pics.append(torch_tensor)
    all_pics.append(cell_feats1)

    dataset = TensorDataset(*all_pics)

    indices = np.arange(cell_num)
    np.random.shuffle(indices)
    train_size = int(cell_num * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    data_loader_test = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
    data_loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    data_loader_train = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=False, shuffle=True)

    # --------------------------
    # Training starts here
    # --------------------------

    # --------------------------
    # Optional pretrain
    # --------------------------
    if pretrain is True:
        optimizer_chr = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        optimizer_patch = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

        # Stage 1: aepretrain
        best_val = float("inf")
        best_state = None
        p_count = 0

        for epoch in range(pretrain_epoch):
            model.train()
            epoch_loss_sum = 0.0

            for inputs in data_loader_train:
                inputs = [item.to(device, non_blocking=True) for item in inputs]
                optimizer_chr.zero_grad(set_to_none=True)
                chr_loss, chr_single_loss = model.aepretrain(inputs)
                loss = chr_loss
                loss.backward()
                optimizer_chr.step()
                epoch_loss_sum += float(loss.item())

            avg_train = epoch_loss_sum / max(1, len(data_loader_train))
            if avg_train < best_val:
                best_val = avg_train
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                p_count = 0
            else:
                p_count += 1
                if p_count > patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
            model.to(device)

        # Stage 2: patch_pretrain
        best_val = float("inf")
        best_state = None
        p_count = 0

        for epoch in range(pretrain_epoch):
            model.train()
            epoch_loss_sum = 0.0

            for inputs in data_loader_train:
                inputs = [item.to(device, non_blocking=True) for item in inputs]
                optimizer_patch.zero_grad(set_to_none=True)
                chr_loss = model.patch_pretrain(inputs)
                chr_loss.backward()
                optimizer_patch.step()
                epoch_loss_sum += float(chr_loss.item())

            avg_train = epoch_loss_sum / max(1, len(data_loader_train))
            if avg_train < best_val:
                best_val = avg_train
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                p_count = 0
            else:
                p_count += 1
                if p_count > patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
            model.to(device)

    elif pretrain is not False:
        w_path = save_name_prefix + str(pretrain) + "dim_" + str(hidden_dim) + ".pth"
        all_states = torch.load(w_path, map_location="cpu")
        model.load_state_dict(all_states['best_pretrain']['model'])
        model.to(device)

    # --------------------------
    # Main training - early stop on VAL loss
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=atlr, weight_decay=weight_decay, betas=betas)
    best_main_ckpt = save_name_prefix + f"best_main_num{num}_hd{hidden_dim}.pt"
    train_history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": None,
        "epochs_ran": 0,
        "early_stop_triggered": False,
    }

    best_val = float("inf")
    best_state = None
    p_count = 0

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        running_loss = 0.0

        for inputs in data_loader_train:
            inputs = [item.to(device, non_blocking=True) for item in inputs]
            optimizer.zero_grad(set_to_none=True)

            x, chr_pred, cell_embed, token_class, patch_loss, chr_loss, chr_single_loss = model(inputs, mask_ratio=mask_ratio)
            loss = loss_ratio * patch_loss + (1 - loss_ratio) * (1 - chr_single_ratio) * chr_loss + (1 - loss_ratio) * (chr_single_ratio) * chr_single_loss

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(1, len(data_loader_train))

        # ---- val ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs in data_loader_val:
                inputs = [item.to(device, non_blocking=True) for item in inputs]
                x, chr_pred, cell_embed, token_class, patch_loss, chr_loss, chr_single_loss = model(inputs, mask_ratio=mask_ratio)
                loss = loss_ratio * patch_loss + (1 - loss_ratio) * (1 - chr_single_ratio) * chr_loss + (1 - loss_ratio) * (chr_single_ratio) * chr_single_loss
                val_loss_sum += float(loss.item())

        avg_val_loss = val_loss_sum / max(1, len(data_loader_val))

        train_history["epochs_ran"] += 1
        train_history["train_loss"].append(avg_train_loss)
        train_history["val_loss"].append(avg_val_loss)

        print(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}")

        # early stop tracking
        improved = avg_val_loss < best_val
        if improved:
            best_val = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "stage": "main_train",
                    "epoch": epoch + 1,
                    "best_metric": best_val,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                },
                best_main_ckpt,
            )
            p_count = 0
        else:
            p_count += 1
            if p_count > patience:
                train_history["early_stop_triggered"] = True
                break

    train_history["best_val_loss"] = best_val

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(device)

    # --------------------------
    # Inference: get embeddings and chr_pred
    # --------------------------
    cell_embeds = []
    chr_preds = []

    model.eval()

    with torch.inference_mode():
        for inputs in data_loader_test:
            inputs = [item.to(device, non_blocking=True) for item in inputs]
            chr_pred, cell_embed = model.inference(inputs)

            cell_embeds.append(cell_embed.cpu())
            chr_preds.append(chr_pred.cpu())

    cell_embed_matrix = torch.cat(cell_embeds, dim=0).numpy()
    chr_pred_matrix = torch.cat(chr_preds, dim=0).numpy()

    # Save embeddings
    embed_out = save_name_prefix + str(num) + "cell_embeddings.npy"
    np.save(embed_out, cell_embed_matrix)

    # Save chr_pred
    chr_pred_out = save_name_prefix + str(num) + "chr_pred.npy"
    np.save(chr_pred_out, chr_pred_matrix)

    print("Training finished.")
    print(f"Best val loss: {train_history['best_val_loss']}")
    print(f"Embeddings saved: {embed_out}")
    print(f"Chr_pred saved: {chr_pred_out}")
