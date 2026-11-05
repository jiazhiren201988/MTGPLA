"""Portable checkpoint loading and evaluation shared by the command-line tools."""
import inspect
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def load_torch(path, device="cpu", trusted_objects=False):
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = not trusted_objects
    return torch.load(path, **kwargs)


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    return torch.device(name)


def load_model(checkpoint, device):
    from model import DeepTTG
    model = DeepTTG().to(device)
    model.load_state_dict(load_torch(checkpoint, device), strict=True)
    return model.eval()


def make_loader(root, phase, batch_size, shuffle=False):
    from dataset import TestbedDataset
    path = Path(root) / "processed" / (phase + ".pt")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; see data/README.md.")
    return DataLoader(TestbedDataset(root=str(root), dataset=phase),
                      batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, targets, predictions = [], [], []
    for batch in loader:
        ids.extend(list(batch.pdbid))
        batch = batch.to(device)
        prediction, _, _ = model(batch)
        targets.append(batch.y.cpu().numpy().reshape(-1))
        predictions.append(prediction.cpu().numpy().reshape(-1))
    return ids, np.concatenate(targets), np.concatenate(predictions)


def scores(targets, predictions):
    import metrics
    return {name: float(function(targets, predictions)) for name, function in (
        ("R", metrics.CORR), ("RMSE", metrics.RMSE), ("MAE", metrics.MAE),
        ("SD", metrics.SD), ("CI", metrics.c_index))}
