import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from train import parse_opt, train
from utils.callbacks import Callbacks
from utils.general import increment_path
from utils.torch_utils import select_device


def sweep():
    wandb.init()
    # Get hyp dict from sweep agent. Copy because train() modifies parameters which confused wandb.
    hyp_dict = vars(wandb.config).get("_items").copy()

    # Workaround: get necessary opt args
    opt = parse_opt(known=True)
    opt.batch_size = hyp_dict.get("batch_size")
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = hyp_dict.get("epochs")
    opt.nosave = True
    opt.data = '/home/selim/Desktop/datasets/hybrid/data.yaml'
    opt.weights = str(opt.weights)
    opt.cfg = str(opt.cfg)
    #opt.hyp = str(hyp_dict.hyp)
    opt.project = str(opt.project)
    opt.img_size = hyp_dict.get("img_size")
    opt.weights = 'yolov5s.pt'
    opt.cfg = 'models/yolov5s_mlt.yaml'
    device = select_device(opt.device, batch_size=opt.batch_size)

    # train
    train(hyp_dict, opt, device, callbacks=Callbacks())


if __name__ == "__main__":
    sweep()
