import os
import time
import copy
import random
import numpy as np
import torch

from ..Model import model_training


def run_training(task):
    run_idx, base_args = task

    args = copy.deepcopy(base_args)
    args.seed = args.seed + run_idx

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    args.lrp_strength_file = os.path.join(
        args.outPath,
        f"LRI_module_strength_run_{run_idx}.txt"
    )

    start = time.time()

    print(f"[INFO] Run {run_idx} starts on {device}")
    print(f"[INFO] Output Background file: {args.lrp_strength_file}")

    model_training.Train_CCC_model_parallel(args)

    duration = time.time() - start
    print(f"[INFO] Finish training run {run_idx}, total time: {duration:.2f}s")

    return args.lrp_strength_file