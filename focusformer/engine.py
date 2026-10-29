"""Trainer and config setup shared by train.py and test.py."""
import copy
import hashlib
import itertools
import logging
import os
import random
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set

import numpy as np
import torch
import yaml

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    HookBase,
    SimpleTrainer,
    TrainerBase,
    create_ddp_model,
    default_setup,
)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from .checkpoint import (
    FOCAL_COORDINATE_VERSION,
    BoundCheckpointer,
    image_manifest_hash,
    initialization_record,
    protect_evaluation_output,
    read_bound_checkpoint,
    resume_checkpoint_path,
)
from .config import add_focusformer_config
from .data import FocalStackClipDatasetMapper, register_instances_focal_stack

# Number of focal planes and intensity statistics of each dataset.
DATASETS = {
    "embryo": {"planes": 11, "pixel_mean": [189.424251], "pixel_std": [51.464221]},
    "cisd": {"planes": 21, "pixel_mean": [185.7824242860772], "pixel_std": [54.480435534073166]},
}


class Trainer(DefaultTrainer):
    """Detectron2 trainer for FocusFormer."""

    def __init__(self, cfg):
        TrainerBase.__init__(self)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = BoundCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            config_text=cfg.dump(),
            trainer=weakref.proxy(self),
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self._hooks: List[HookBase] = []
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        if not resume:
            if self.checkpointer.has_checkpoint():
                raise ValueError("Existing checkpoint in output directory; use --resume or a fresh directory")
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=False)
            self.checkpointer.initialization = initialization_record(self.cfg, self.checkpointer.path_manager)
            return
        checkpoint = read_bound_checkpoint(
            resume_checkpoint_path(self.cfg.OUTPUT_DIR), self.cfg.dump(), resume=True
        )
        self.checkpointer.initialization = checkpoint.get("initialization", {"status": "not recorded"})
        self.checkpointer.model.load_state_dict(checkpoint["model"], strict=True)
        self.load_state_dict(checkpoint["trainer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_iter = checkpoint["iteration"] + 1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = FocalStackClipDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = FocalStackClipDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def build_writers(self):
        from detectron2.utils.events import CommonMetricPrinter, JSONWriter
        return [
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """Evaluate a model that already holds its trained weights."""
        logger = logging.getLogger(__name__)
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)
            with torch.autocast("cuda", enabled=bool(cfg.SOLVER.AMP.ENABLED)):
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        if len(results) == 1:
            results = list(results.values())[0]
        return results


def strict_load_training_checkpoint(model, checkpoint_path, config_text):
    """Do not silently skip missing, unexpected, or differently shaped tensors."""
    checkpoint = read_bound_checkpoint(checkpoint_path, config_text)
    model.load_state_dict(checkpoint["model"], strict=True)


def _base_cfg():
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_focusformer_config(cfg)
    return cfg


def _check_cfg(cfg, dataset):
    if dataset not in DATASETS:
        raise ValueError("Dataset must be embryo or cisd")
    expected_planes = DATASETS[dataset]["planes"]
    if (cfg.INPUT.SAMPLING_FRAME_NUM != expected_planes
            or not cfg.INPUT.USE_FULL_FOCAL_STACK
            or cfg.INPUT.SAMPLING_STRATEGY != "full"
            or cfg.INPUT.SAMPLING_FRAME_SHUFFLE):
        raise ValueError(f"{dataset} requires its full ordered {expected_planes}-plane stack")
    if cfg.DATASETS.DATA_TYPE != dataset:
        raise ValueError("Dataset configuration mismatch")
    if cfg.INPUT.FOCAL_COORDINATE_VERSION != FOCAL_COORDINATE_VERSION:
        raise ValueError("Unsupported focal coordinate version")
    if cfg.SOLVER.REFERENCE_WORLD_SIZE != 0:
        raise ValueError("Automatic worker scaling is not supported")
    if cfg.MODEL.MASK_FORMER.USE_GAUSSIAN_FOCUS != (cfg.MODEL.FUSE_STRATEGY == "gaussian"):
        raise ValueError("Gaussian module and fusion configuration disagree")


def _finish_setup(cfg, args, name, annotation, image_root):
    cfg.MODEL.DEVICE = f"cuda:{comm.get_local_rank()}" if torch.cuda.is_available() else "cpu"
    cfg.TEST.EVAL_PERIOD = 0
    cfg.OUTPUT_DIR = args.output_dir
    register_instances_focal_stack(name, {}, annotation, image_root)
    if cfg.SEED >= 0:
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="")
    return cfg


def setup_train(args):
    """Build the training config from a config file and the chosen dataset/fold."""
    dataset = args.dataset
    cfg = _base_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.DATA_TYPE = dataset
    cfg.MODEL.PIXEL_MEAN = DATASETS[dataset]["pixel_mean"]
    cfg.MODEL.PIXEL_STD = DATASETS[dataset]["pixel_std"]
    cfg.INPUT.SAMPLING_FRAME_NUM = DATASETS[dataset]["planes"]
    cfg.merge_from_list(args.opts)
    _check_cfg(cfg, dataset)

    split_dir = args.split_dir or os.path.join(args.data_root, dataset, "annotation", f"split_{args.split}")
    annotation = os.path.join(split_dir, "train.json")
    image_root = os.path.join(args.data_root, dataset, "images")
    if not os.path.isfile(annotation):
        raise FileNotFoundError(annotation)
    with open(annotation, "rb") as handle:
        cfg.DATASETS.TRAIN_ANNOTATION_SHA256 = hashlib.sha256(handle.read()).hexdigest()
    cfg.DATASETS.SPLIT_DIR = os.path.realpath(split_dir)
    cfg.DATASETS.IMAGE_ROOT = os.path.realpath(image_root)
    cfg.DATASETS.IMAGE_MANIFEST_SHA256 = image_manifest_hash(annotation, cfg.DATASETS.IMAGE_ROOT)
    cfg.DATASETS.TRAIN = ("focal_stack_train",)
    cfg.DATASETS.TEST = ()

    if args.resume:
        previous = read_bound_checkpoint(resume_checkpoint_path(args.output_dir), cfg.dump(), resume=True)
        previous_root = yaml.safe_load(previous["training_config"]["config_yaml"]).get("DATASETS", {}).get("IMAGE_ROOT", "")
        if previous_root != cfg.DATASETS.IMAGE_ROOT:
            logging.getLogger(__name__).warning("Resume image root differs from the recorded training image root")
    elif (os.path.isfile(os.path.join(args.output_dir, "config.yaml"))
          or os.path.isfile(os.path.join(args.output_dir, "last_checkpoint"))):
        raise ValueError("New training requires a fresh output directory; do not overwrite an existing run")
    return _finish_setup(cfg, args, "focal_stack_train", annotation, image_root)


def setup_test(args):
    """Rebuild the config saved with a trained checkpoint for evaluation on the test split."""
    protect_evaluation_output(args.output_dir)
    checkpoint_path = os.path.realpath(os.path.expanduser(args.weights))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml saved by training must be beside the checkpoint: {config_path}")
    with open(config_path) as handle:
        saved = yaml.safe_load(handle)
    read_bound_checkpoint(checkpoint_path, yaml.safe_dump(saved))
    dataset = saved["DATASETS"]["DATA_TYPE"]
    if os.path.realpath(args.output_dir) == os.path.dirname(checkpoint_path):
        raise ValueError("Evaluation output must not overwrite the training directory")

    cfg = _base_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = checkpoint_path
    if hasattr(cfg.MODEL, "HIERA"):
        # The trained checkpoint already contains the encoder weights.
        cfg.MODEL.HIERA.LOAD_PRETRAINED = False
    _check_cfg(cfg, dataset)

    split_dir = args.split_dir or os.path.join(args.data_root, dataset, "annotation", f"split_{args.split}")
    annotation = os.path.join(split_dir, "test.json")
    image_root = os.path.join(args.data_root, dataset, "images")
    if not os.path.isfile(annotation):
        raise FileNotFoundError(annotation)
    if getattr(cfg.DATASETS, "SPLIT_DIR", "") != os.path.realpath(split_dir):
        logging.getLogger(__name__).warning("Evaluation split path differs from the recorded training split")
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ("focal_stack_test",)
    return _finish_setup(cfg, args, "focal_stack_test", annotation, image_root)
