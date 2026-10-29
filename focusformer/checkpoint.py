"""Bind trained weights to their configuration and require complete resume state."""
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml
from detectron2.checkpoint import DetectionCheckpointer

FOCAL_COORDINATE_VERSION = "full_stack_voxel_centers_v1"
CONTRACT_VERSION = 2


def semantic_config(config_text):
    config = copy.deepcopy(yaml.safe_load(config_text))
    for key in ("OUTPUT_DIR", "VIS_PERIOD"):
        config.pop(key, None)
    config.get("DATALOADER", {}).pop("NUM_WORKERS", None)
    config.get("SOLVER", {}).pop("CHECKPOINT_PERIOD", None)
    model = config.get("MODEL", {})
    for key in ("DEVICE", "WEIGHTS"):
        model.pop(key, None)
    for key in ("CHECKPOINT", "LOAD_PRETRAINED"):
        model.get("HIERA", {}).pop(key, None)
    datasets = config.get("DATASETS", {})
    for key in ("TRAIN", "TEST", "SPLIT_DIR", "IMAGE_ROOT"):
        datasets.pop(key, None)
    return config


def config_hash(config_text):
    canonical = json.dumps(semantic_config(config_text), sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def binding(config_text):
    return {"version": CONTRACT_VERSION, "config_yaml": config_text, "sha256": config_hash(config_text)}


def different_keys(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(name)
            else:
                result.extend(different_keys(left[key], right[key], name))
        return result
    return [] if left == right else [prefix]


def protect_evaluation_output(output_dir):
    folder = Path(output_dir)
    if (folder / "config.yaml").exists() or (folder / "last_checkpoint").exists() or any(folder.glob("model_*.pth")):
        raise ValueError("Evaluation output contains training/config artifacts; choose a fresh output directory")


def image_manifest_hash(annotation_path, image_root):
    """Fingerprint referenced .npy names/sizes/mtimes, not their file contents."""
    with open(annotation_path) as handle:
        images = json.load(handle)["images"]
    records = []
    for image in images:
        name = image["file_name"]
        if not name.endswith(".npy"):
            name = name[:-4] + ".npy"
        stat = (Path(image_root) / name).stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    encoded = json.dumps(sorted(records), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def initialization_record(cfg, path_manager=None):
    sources = {"model_weights": cfg.MODEL.WEIGHTS}
    if cfg.MODEL.BACKBONE.NAME == "SAM2HieraSBackbone":
        sources["hiera_pretrained"] = cfg.MODEL.HIERA.CHECKPOINT
    records = {}
    for name, source in sources.items():
        record = {"source": source, "sha256": None}
        path = Path(source).expanduser() if source else None
        if source and not path.is_file() and path_manager is not None:
            path = Path(path_manager.get_local_path(source))
        if path is not None and path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            record.update({"resolved_path": str(path.resolve()), "sha256": digest.hexdigest()})
        records[name] = record
    return records


def read_bound_checkpoint(path, config_text=None, resume=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Expected a trained model checkpoint")
    contract = checkpoint.get("training_config", {})
    if not isinstance(contract, dict):
        raise ValueError("Invalid checkpoint training configuration")
    if contract.get("version") != CONTRACT_VERSION:
        raise ValueError("Unsupported checkpoint contract version")
    text = contract.get("config_yaml")
    if not isinstance(text, str) or contract.get("sha256") != config_hash(text):
        raise ValueError("Checkpoint has no valid embedded training configuration")
    saved = yaml.safe_load(text)
    if saved.get("INPUT", {}).get("FOCAL_COORDINATE_VERSION") != FOCAL_COORDINATE_VERSION:
        raise ValueError("Checkpoint focal-coordinate version is missing or incompatible")
    if config_text is not None and config_hash(config_text) != contract["sha256"]:
        keys = different_keys(semantic_config(text), semantic_config(config_text))
        raise ValueError("Configuration differs from checkpoint: " + ", ".join(keys))
    if type(checkpoint.get("iteration")) is not int:
        raise ValueError("Checkpoint has no valid training iteration")
    if resume:
        trainer = checkpoint.get("trainer", {})
        inner = trainer.get("_trainer", {})
        optimizer = inner.get("optimizer", {})
        if (trainer.get("iteration") != checkpoint["iteration"]
                or not isinstance(optimizer, dict)
                or "state" not in optimizer or "param_groups" not in optimizer
                or not isinstance(checkpoint.get("scheduler"), dict)
                or not checkpoint["scheduler"]):
            raise ValueError("Resume requires complete trainer, optimizer, scheduler and consistent iteration state; model-only weights are not resumable")
        if saved.get("SOLVER", {}).get("AMP", {}).get("ENABLED", False):
            scaler = inner.get("grad_scaler", {})
            required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
            if not isinstance(scaler, dict) or not required.issubset(scaler):
                raise ValueError("AMP resume requires GradScaler state")
    return checkpoint


def resume_checkpoint_path(output_dir):
    marker = Path(output_dir) / "last_checkpoint"
    if not marker.is_file():
        raise FileNotFoundError("--resume requires last_checkpoint; use a fresh output directory for a new run")
    value = marker.read_text().strip()
    if not value:
        raise ValueError("Empty last_checkpoint")
    path = Path(value)
    path = path if path.is_absolute() else Path(output_dir) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class BoundCheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir, *, config_text, **checkpointables):
        super().__init__(model, save_dir, **checkpointables)
        self.training_binding = binding(config_text)
        self.initialization = {"status": "not recorded"}

    def save(self, name, **kwargs):
        kwargs["training_config"] = self.training_binding
        kwargs["initialization"] = self.initialization
        super().save(name, **kwargs)
