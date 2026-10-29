"""Evaluate a trained FocusFormer checkpoint on the test split of its fold."""
import warnings

import torch
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch

from focusformer.engine import Trainer, setup_test, strict_load_training_checkpoint

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())
    cfg = setup_test(args)
    model = Trainer.build_model(cfg)
    strict_load_training_checkpoint(model, cfg.MODEL.WEIGHTS, cfg.dump())
    return Trainer.test(cfg, model)


def build_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--weights", required=True, help="Trained checkpoint; its config.yaml must be in the same directory")
    parser.add_argument("--split", type=int, default=1, help="Cross-validation fold (1-5)")
    parser.add_argument("--data_root", default="data", help="Directory containing embryo/ and cisd/")
    parser.add_argument("--split_dir", default="", help="Annotation directory containing test.json (overrides --split)")
    parser.add_argument("--output_dir", default="work_dirs/eval", help="Output directory for predictions and metrics")
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
