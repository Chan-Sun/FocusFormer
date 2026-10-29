"""Train FocusFormer on one fold of a focal-stack dataset."""
import warnings

import torch
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch

from focusformer.engine import Trainer, setup_train

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())
    cfg = setup_train(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def build_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--dataset", choices=("embryo", "cisd"), required=True, help="Dataset to train on")
    parser.add_argument("--split", type=int, default=1, help="Cross-validation fold (1-5)")
    parser.add_argument("--data_root", default="data", help="Directory containing embryo/ and cisd/")
    parser.add_argument("--split_dir", default="", help="Annotation directory containing train.json (overrides --split)")
    parser.add_argument("--output_dir", default="work_dirs/debug", help="Output directory")
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    if len(args.opts) % 2:
        parser.error("Config overrides must be KEY VALUE pairs placed after all --flags")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
