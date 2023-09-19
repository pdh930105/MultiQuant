import logging
import os
import socket
import hydra

logger = logging.getLogger(__name__)
rank = 0
local_rank = 0
world_size = 1

def run(args):
    import src.distrib as distrib
    import src.dataset as dataset
    from trainer import Trainer
    import torch
    import torch.nn as nn

    # Set up distributed training
    logger.info("Running on host %s", socket.gethostname())
    distrib.init(args, args.rendezvous_file)
    torch.manual_seed(args.seed)

    assert args.batch_size % distrib.get_world_size() == 0, "Batch size must be divisible by number of GPUs"
    
    args.batch_size = args.batch_size // distrib.world_size

    model = build_model(args.arch)
    img_size = args.img_size

    trainset, testset, num_classes = dataset.get_loader(args, img_size)

    tr_loader = distrib.loader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    tt_loader = distrib.loader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    data = {"tr": tr_loader, "tt": tt_loader}
    
    logger.debug(model)


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.continue_from:
        args.continue_from = os.path.join(os.getcwd(), "..", args.continue_from, args.checkpoint_file)
    args.db.root = hydra.utils.to_absolute_path(args.db.root)

    logger.debug(args)
    run(args)

@hydra.main(config_path="conf", config_name="config")
def main(args):
    try:
        if args.ddp and args.rank is None:
            from src.executor import start_ddp_workers
            start_ddp_workers(args)
            return
        _main(args)
    except Exception:
        logger.exception("Unexpected error")
        os._exit(1)