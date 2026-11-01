import argparse
import json
import math
import pickle
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .model import hicformer
from .utils import BandNorm


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, embedding):
        return self.fc2(self.relu(self.fc1(embedding)))


def load_encoder_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "best_pretrain" in checkpoint:
        state = checkpoint["best_pretrain"]["model"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    prefixes = {"patch_embed_branches.": "multi_encoder.", "blocks.": "transformer_blocks.",
                "whole_pred.": "cell_decoder.", "patch_pred.": "block_decoder.",
                "chr_pred.": "chr_decoder."}
    mapped = {}
    for name, value in state.items():
        for source, target in prefixes.items():
            if name.startswith(source):
                name = target + name[len(source):]
                break
        mapped[name] = value
    return mapped


def prepare_data(cfg):
    with open(cfg["label_path"], "rb") as handle:
        labels = np.asarray(pickle.load(handle)[cfg["label_name"]])
    classes, encoded = np.unique(labels, return_inverse=True)
    pictures, components = [], []
    for chromosome in cfg["chr_list"]:
        path = Path(cfg["raw_path"]) / f"{chromosome}_sparse_adj.npy"
        matrices = np.load(path, allow_pickle=True)
        if len(matrices) != len(labels):
            raise ValueError(f"Expected one label per cell in {path}")
        dense = [matrix.toarray() for matrix in matrices]
        if cfg["band_norm"]:
            dense = BandNorm(dense, ifdiagonal=False)
        flat = np.asarray(dense).reshape(len(labels), -1)
        tokens = PCA(n_components=cfg["embed_dim"], random_state=3).fit_transform(flat)
        if cfg["enorm"]:
            tokens = StandardScaler().fit_transform(tokens)
        components.append(torch.from_numpy(tokens.astype(np.float32, copy=False)))
        pictures.append(torch.stack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in dense]))
        print(f"Prepared {chromosome}", flush=True)
    scale = torch.zeros(len(labels), 1)
    for cell in range(len(labels)):
        total, count = 0.0, 0
        for chromosome in pictures:
            positive = chromosome[cell][chromosome[cell] > 0]
            total += positive.sum().item()
            count += positive.numel()
        if count:
            scale[cell, 0] = total / count
    whole_size = sum(x.shape[-1] * (x.shape[-1] + 1) // 2 for x in pictures)
    dataset = TensorDataset(*pictures, torch.stack(components, dim=1), scale,
                            torch.tensor(encoded, dtype=torch.long))
    return dataset, classes.tolist(), labels.tolist(), whole_size


def run_epoch(model, classifier, loader, cfg, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    classifier.train(training)
    totals = np.zeros(3)
    labels, predictions = [], []
    batches = cells = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = [x.to(device) for x in batch[:-1]]
            target = batch[-1].to(device).reshape(-1)
            if training:
                optimizer.zero_grad()
            _, _, embedding, _, patch, chromosome, single = model(inputs, mask_ratio=cfg["mask_ratio"])
            logits = classifier(embedding)
            ce = nn.functional.cross_entropy(logits, target)
            r, s = cfg["loss_ratio"], cfg["chr_single_ratio"]
            loss = r * patch + (1 - r) * (1 - s) * chromosome + (1 - r) * s * single + cfg["class_ratio"] * ce
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            totals += np.asarray([loss.item(), chromosome.item(), ce.item()]) * target.numel()
            cells += target.numel()
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            batches += 1
    totals /= cells
    metrics = {"accuracy": float(accuracy_score(labels, predictions)),
               "kappa": float(cohen_kappa_score(labels, predictions)),
               "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0))}
    return {"loss": totals[0], "chr_loss": totals[1], "ce_loss": totals[2],
            "metrics": metrics, "labels": labels, "predictions": predictions,
            "batch_count": batches}


def evaluate(model, classifier, loader, cfg, device, seed):
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        return run_epoch(model, classifier, loader, cfg, device)


def main():
    parser = argparse.ArgumentParser(description="Joint Hi-Cformer and cell-type classifier training.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/classifier.json")
    parser.add_argument("--checkpoint", help="Pretrained encoder checkpoint")
    parser.add_argument("--output", help="New output directory")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with open(args.config) as handle:
        settings = json.load(handle)
    if args.dataset not in settings["datasets"]:
        parser.error(f"Unknown dataset: {args.dataset}")
    cfg = {**settings["defaults"], **settings["datasets"][args.dataset]}
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if not 0 < cfg["validation_fraction"] < 1:
        parser.error("validation_fraction must be between zero and one")
    if any(type(cfg[key]) is not int or cfg[key] < 1 for key in ["max_epochs", "patience"]):
        parser.error("max_epochs and patience must be positive integers")
    if not math.isfinite(cfg["min_delta"]) or cfg["min_delta"] < 0:
        parser.error("min_delta must be nonnegative and finite")
    if not Path(cfg["checkpoint"]).is_file():
        parser.error(f"Checkpoint not found: {cfg['checkpoint']}; supply --checkpoint")
    root = Path(args.output or f"experiments/classifier/{args.dataset}")
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(cfg, indent=2))
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = cfg["cudnn_benchmark"]

    dataset, classes, labels, whole_size = prepare_data(cfg)
    state = load_encoder_state(cfg["checkpoint"])
    model_config = {key: cfg[key] for key in ["img_size", "patch_size", "encoder_dim", "hidden_dim",
                                            "embed_dim", "depth", "weight", "chr_single_loss", "otherlf", "enorm"]}
    model_config.update(whole_size=whole_size, chr_num=len(cfg["chr_list"]), in_chans=1,
                        num_heads=8, mlp_ratio=4., chr_mutual_visibility=cfg["vis"],
                        num_patches=state["pos_embed.weight"].shape[0] - 1, decoder_activation="gelu")
    model = hicformer(**model_config, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    classifier = Classifier(cfg["hidden_dim"], len(classes))
    model.to(args.device)
    classifier.to(args.device)
    initial_classifier = {name: value.detach().cpu().clone() for name, value in classifier.state_dict().items()}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    records = []
    targets = dataset.tensors[-1].numpy()
    for fold, (outer_train, heldout) in enumerate(splitter.split(np.zeros(len(labels)), targets)):
        fold_seed = cfg["seed"] + fold
        train, validation = train_test_split(outer_train, test_size=cfg["validation_fraction"],
                                             stratify=targets[outer_train], random_state=fold_seed)
        if len(np.unique(targets[train])) != len(classes) or len(np.unique(targets[validation])) != len(classes):
            raise ValueError("Every class must occur in training and validation; adjust validation_fraction or add cells")
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
        random.seed(fold_seed)
        model.load_state_dict(state, strict=True)
        classifier.load_state_dict(initial_classifier, strict=True)
        train_loader = DataLoader(Subset(dataset, train), batch_size=cfg["batch_size"], shuffle=True)
        validation_loader = DataLoader(Subset(dataset, validation), batch_size=cfg["batch_size"], shuffle=False)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()),
                                     lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
        updates = bad_epochs = best_epoch = best_updates = 0
        best_ce = patience_reference = float("inf")
        validation_seed, test_seed = 1000 + fold_seed, 2000 + fold_seed
        checkpoint_path = root / f"fold{fold}.pth"
        for epoch in range(1, cfg["max_epochs"] + 1):
            trained = run_epoch(model, classifier, train_loader, cfg, args.device, optimizer)
            updates += trained["batch_count"]
            validated = evaluate(model, classifier, validation_loader, cfg, args.device, validation_seed)
            if validated["ce_loss"] < best_ce:
                best_ce, best_epoch, best_updates = validated["ce_loss"], epoch, updates
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save({"model_state_dict": model.state_dict(), "classifier_state_dict": classifier.state_dict(),
                            "model_config": model_config, "classes": classes, "config": cfg,
                            "fold": fold, "epoch": best_epoch, "updates": best_updates,
                            "validation_ce": best_ce}, temporary)
                temporary.replace(checkpoint_path)
            if validated["ce_loss"] < patience_reference - cfg["min_delta"]:
                patience_reference, bad_epochs = validated["ce_loss"], 0
            else:
                bad_epochs += 1
            event = {"event": "train_epoch", "fold": fold, "epoch": epoch, "updates": updates,
                     "train_loss": trained["loss"], "train_ce": trained["ce_loss"],
                     "validation_ce": validated["ce_loss"], "best_epoch": best_epoch, "bad_epochs": bad_epochs}
            with (root / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            if bad_epochs >= cfg["patience"]:
                break
        selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model_state_dict"], strict=True)
        classifier.load_state_dict(selected["classifier_state_dict"], strict=True)
        del selected
        heldout_loader = DataLoader(Subset(dataset, heldout), batch_size=cfg["batch_size"], shuffle=False)
        evaluated = evaluate(model, classifier, heldout_loader, cfg, args.device, test_seed)
        record = {"fold": fold, "fold_seed": fold_seed, "validation_seed": validation_seed, "test_seed": test_seed,
                  "epochs_trained": epoch, "best_epoch": best_epoch, "optimizer_updates": updates,
                  "checkpoint_updates": best_updates, "validation_ce": best_ce,
                  "train_indices": train.tolist(), "validation_indices": validation.tolist(),
                  "heldout_indices": heldout.tolist(), **evaluated}
        records.append(record)
        (root / f"fold{fold}_predictions.json").write_text(json.dumps(record, indent=2))
        means = {metric: float(np.mean([r["metrics"][metric] for r in records]))
                 for metric in ["accuracy", "kappa", "macro_f1"]}
        (root / "results.json").write_text(json.dumps({"classes": classes, "folds": records, "mean_metrics": means}, indent=2))
        print(json.dumps({"event": "fold_completed", "fold": fold, "optimizer_updates": updates,
                          "metrics": evaluated["metrics"]}), flush=True)
    print(json.dumps({"mean_metrics": means}), flush=True)


if __name__ == "__main__":
    main()
