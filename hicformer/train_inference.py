import argparse
import json
import os
import pickle
from functools import partial

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import BandNorm, cal_expand_tensor
from .model import hicformer


# --------------------------
# Unified config defaults
# --------------------------
DEFAULT_CONFIG = {
    "label_path": "data/Ramani2017/label_info.pickle",
    "label_name": "cell type",
    "raw_path": "data/Ramani2017/raw",
    "save_path": "experiments/default",
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
    "shuffle_before_split": True,
    "mask_ratio": 0.5,
    "atlr": 5e-4,
    "otherlf": "MSE",
    "enorm": "False",
    "seed": 1,
    "pca_method": "truncated_svd",
    "pca_seed": 3,
    "num_workers": 4,
    "band_norm": False,
    "band_norm_diagonal": False,
    "checkpoint_metric": "val_total",
    "restore_stage1_best": True,
    "save_chr_pred": True,
    "decoder_activation": "none",
    "embedding_activation": "none",
    "cudnn_benchmark": False,
}


def load_config(config_path):
    cfg = DEFAULT_CONFIG.copy()
    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        # Historical configs used the human-readable key "learning rate".
        if "learning rate" in user_cfg and "lr" not in user_cfg:
            user_cfg["lr"] = user_cfg["learning rate"]
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
    parser.add_argument('--shuffle_before_split', type=str, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--atlr', type=float, default=None)
    parser.add_argument('--otherlf', type=str, default=None)
    parser.add_argument('--enorm', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pca_method', choices=['pca', 'truncated_svd'], default=None)
    parser.add_argument('--pca_seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--band_norm', type=str, default=None)
    parser.add_argument('--band_norm_diagonal', type=str, default=None)
    parser.add_argument(
        '--checkpoint_metric',
        choices=['val_total', 'train_total', 'train_chr'],
        default=None,
    )
    parser.add_argument('--restore_stage1_best', type=str, default=None)
    parser.add_argument('--save_chr_pred', type=str, default=None)
    parser.add_argument('--decoder_activation', choices=['none', 'gelu'], default=None)
    parser.add_argument('--embedding_activation', choices=['none', 'gelu'], default=None)
    parser.add_argument('--cudnn_benchmark', type=str, default=None)

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
    os.makedirs(save_path, exist_ok=True)
    prefix_parent = os.path.dirname(save_name_prefix)
    if prefix_parent:
        os.makedirs(prefix_parent, exist_ok=True)

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
        raise ValueError("pretrain must be true or false")

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
    shuffle_before_split = str(cfg["shuffle_before_split"]).lower() == "true"
    mask_ratio = cfg["mask_ratio"]
    atlr = cfg["atlr"]
    seed = int(cfg["seed"])
    pca_method = cfg["pca_method"]
    pca_seed = int(cfg["pca_seed"])
    num_workers = int(cfg["num_workers"])
    band_norm = str(cfg["band_norm"]).lower() == "true"
    band_norm_diagonal = str(cfg["band_norm_diagonal"]).lower() == "true"
    checkpoint_metric = cfg["checkpoint_metric"]
    restore_stage1_best = str(cfg["restore_stage1_best"]).lower() == "true"
    save_chr_pred = str(cfg["save_chr_pred"]).lower() == "true"
    cudnn_benchmark = str(cfg["cudnn_benchmark"]).lower() == "true"

    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = cudnn_benchmark
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg["enorm"] = enorm
    cfg["pretrain"] = pretrain
    cfg["vis"] = vis
    cfg["chr_single_loss"] = chr_single_loss_flag
    cfg["seed"] = seed
    cfg["pca_method"] = pca_method
    cfg["pca_seed"] = pca_seed
    cfg["num_workers"] = num_workers
    cfg["band_norm"] = band_norm
    cfg["band_norm_diagonal"] = band_norm_diagonal
    cfg["checkpoint_metric"] = checkpoint_metric
    cfg["restore_stage1_best"] = restore_stage1_best
    cfg["save_chr_pred"] = save_chr_pred
    cfg["cudnn_benchmark"] = cudnn_benchmark
    cfg["shuffle_before_split"] = shuffle_before_split
    with open(os.path.join(save_path, "config.resolved.json"), "w") as handle:
        json.dump(cfg, handle, indent=2)

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
    pca_save_dir = os.path.join(
        save_path,
        "pca_cache",
        (
            f"{pca_method}_dim{embed_dim}_seed{pca_seed}"
            f"_bandnorm{int(band_norm)}_diag{int(band_norm_diagonal)}"
        ),
    )
    os.makedirs(pca_save_dir, exist_ok=True)

    for chrname in chr_list:
        f = folder_path + '/' + chrname + '_sparse_adj.npy'
        loaded_data = np.load(f, allow_pickle=True)
        if len(loaded_data) != len(labels):
            raise ValueError(
                f"{f} contains {len(loaded_data)} cells but labels contain {len(labels)}"
            )
        onechr_pic = [loaded_data[i].toarray() for i in range(len(labels))]
        if band_norm:
            # Match the Lee training protocol: BandNorm defines the matrix
            # representation used by both PCA tokens and reconstruction.
            onechr_pic = BandNorm(
                onechr_pic,
                ifdiagonal=band_norm_diagonal,
            )
        datas_np = np.array(onechr_pic)
        if datas_np.ndim != 3 or datas_np.shape[1] != datas_np.shape[2]:
            raise ValueError(f"{f} has invalid dense shape {datas_np.shape}")

        if datas_np.shape[1] > max_height:
            max_height = datas_np.shape[1]

        datas_flat = datas_np.reshape(datas_np.shape[0], -1)

        onechr_pic_t = torch.stack([
            torch.tensor(pic, dtype=torch.float32).unsqueeze(0)
            for pic in onechr_pic
        ])
        all_pics.append(onechr_pic_t)

        pca_file = os.path.join(pca_save_dir, f"{chrname}_tokens.npy")
        if os.path.exists(pca_file):
            result = np.load(pca_file)
            expected_shape = (len(labels), embed_dim)
            if result.shape != expected_shape:
                raise ValueError(
                    f"cached token shape {result.shape} at {pca_file}; expected {expected_shape}"
                )
        else:
            if pca_method == "pca":
                pca = PCA(n_components=embed_dim, random_state=pca_seed)
            elif pca_method == "truncated_svd":
                pca = TruncatedSVD(n_components=embed_dim, random_state=pca_seed)
            else:
                raise ValueError(f"unsupported pca_method: {pca_method}")
            result = pca.fit_transform(datas_flat)
            result = result.astype(np.float32)
            np.save(pca_file, result)

        if enorm:
            scaler = StandardScaler()
            result = scaler.fit_transform(result)

        chr_cls.append(result.astype(np.float32, copy=False))
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
        otherlf=otherlf, enorm=enorm,
        decoder_activation=cfg["decoder_activation"]
    )
    model.to(device)

    # Original cell-specific reconstruction scale: mean positive contact value.
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

    all_pics.append(torch_tensor)
    all_pics.append(cell_feats1)

    dataset = TensorDataset(*all_pics)

    indices = np.arange(cell_num)
    # Keep this explicit: a fixed pre-permutation changes the seeded DataLoader
    # batch sequence even when train_ratio=1.0.
    if shuffle_before_split:
        np.random.shuffle(indices)
    train_size = min(cell_num, int(cell_num * train_ratio))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    if len(train_indices) == 0:
        raise ValueError(
            f"train_ratio={train_ratio} produced train/val sizes "
            f"{len(train_indices)}/{len(val_indices)}"
        )
    if checkpoint_metric == "val_total" and len(val_indices) == 0:
        raise ValueError("checkpoint_metric=val_total requires a non-empty validation split")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = (
        torch.utils.data.Subset(dataset, val_indices) if len(val_indices) else None
    )

    data_loader_test = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    data_loader_val = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )
    data_loader_train = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, shuffle=True)

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

        if best_state is not None and restore_stage1_best:
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

    # --------------------------
    # Main training - early stop on VAL loss
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=atlr, weight_decay=weight_decay, betas=betas)
    best_main_ckpt = save_name_prefix + f"best_main_num{num}_hd{hidden_dim}.pt"
    train_history = {
        "train_loss": [],
        "train_chr_loss": [],
        "val_loss": [],
        "best_val_loss": None,
        "checkpoint_metric": checkpoint_metric,
        "best_selection_value": None,
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
        running_chr_loss = 0.0

        for inputs in data_loader_train:
            inputs = [item.to(device, non_blocking=True) for item in inputs]
            optimizer.zero_grad(set_to_none=True)

            x, chr_pred, cell_embed, token_class, patch_loss, chr_loss, chr_single_loss = model(inputs, mask_ratio=mask_ratio)
            loss = loss_ratio * patch_loss + (1 - loss_ratio) * (1 - chr_single_ratio) * chr_loss + (1 - loss_ratio) * (chr_single_ratio) * chr_single_loss

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_chr_loss += float(chr_loss.item())

        avg_train_loss = running_loss / max(1, len(data_loader_train))
        avg_train_chr_loss = running_chr_loss / max(1, len(data_loader_train))

        # ---- val ----
        avg_val_loss = None
        if data_loader_val is not None:
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
        train_history["train_chr_loss"].append(avg_train_chr_loss)
        train_history["val_loss"].append(avg_val_loss)

        val_text = "n/a" if avg_val_loss is None else f"{avg_val_loss:.6f}"
        print(
            f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train_loss:.6f} "
            f"train_chr_loss={avg_train_chr_loss:.6f} val_loss={val_text}"
        )

        # early stop tracking
        selection_values = {
            "val_total": avg_val_loss,
            "train_total": avg_train_loss,
            "train_chr": avg_train_chr_loss,
        }
        selection_value = selection_values[checkpoint_metric]
        improved = selection_value < best_val
        if improved:
            best_val = selection_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "stage": "main_train",
                    "epoch": epoch + 1,
                    "best_metric": best_val,
                    "selection_metric": checkpoint_metric,
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

    train_history["best_selection_value"] = (
        best_val if train_history["epochs_ran"] else None
    )
    if checkpoint_metric == "val_total" and train_history["epochs_ran"]:
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
            if cfg["embedding_activation"] == "gelu":
                cell_embed = torch.nn.functional.gelu(cell_embed)

            cell_embeds.append(cell_embed.cpu())
            if save_chr_pred:
                chr_preds.append(chr_pred.cpu())

    cell_embed_matrix = torch.cat(cell_embeds, dim=0).numpy()

    # Save embeddings
    embed_out = save_name_prefix + str(num) + "cell_embeddings.npy"
    np.save(embed_out, cell_embed_matrix)

    # Save chr_pred
    chr_pred_out = None
    if save_chr_pred:
        chr_pred_matrix = torch.cat(chr_preds, dim=0).numpy()
        chr_pred_out = save_name_prefix + str(num) + "chr_pred.npy"
        np.save(chr_pred_out, chr_pred_matrix)

    with open(os.path.join(save_path, "training_history.json"), "w") as handle:
        json.dump(train_history, handle, indent=2)

    print("Training finished.")
    print(f"Best {checkpoint_metric}: {train_history['best_selection_value']}")
    print(f"Embeddings saved: {embed_out}")
    print(f"Chr_pred saved: {chr_pred_out}" if save_chr_pred else "Chr_pred saving disabled.")
