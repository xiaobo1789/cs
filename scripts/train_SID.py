"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_SID import Coach

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # 增大显存分配块大小，减少碎片1
# 在 train_SID.py 或 coach_SID.py 中添加以下代码
print(f"Current working directory: {os.getcwd()}")
train_source_root = "./datasets/train/low"
train_target_root = "./datasets/train/high"
test_source_root = "./datasets/test/low"
test_target_root = "./datasets/test/high"

print(f"实际训练集低光照路径： {os.path.abspath(train_source_root)}")
print(f"实际训练集正常光照路径： {os.path.abspath(train_target_root)}")
print(f"实际测试集低光照路径： {os.path.abspath(test_source_root)}")
print(f"实际测试集正常光照路径： {os.path.abspath(test_target_root)}")

# 检查路径是否存在
if not os.path.exists(train_source_root):
    print(f"训练集低光照路径 {train_source_root} 不存在")
if not os.path.exists(train_target_root):
    print(f"训练集正常光照路径 {train_target_root} 不存在")
if not os.path.exists(test_source_root):
    print(f"测试集低光照路径 {test_source_root} 不存在")
if not os.path.exists(test_target_root):
    print(f"测试集正常光照路径 {test_target_root} 不存在")
# 在 train_SID.py 或 coach_SID.py 中添加以下代码

def main():
    opts = TrainOptions().parse()
    previous_train_ckpt = None

    coach = Coach(opts, previous_train_ckpt)
    coach.train()


def load_train_checkpoint(opts):
    train_ckpt_path = opts.resume_training_from_ckpt
    device_id = torch.cuda.current_device()
    previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location=lambda storage, loc: storage.cuda(device_id))
    new_opts_dict = vars(opts)
    opts = previous_train_ckpt['opts']
    opts['resume_training_from_ckpt'] = train_ckpt_path
    update_new_configs(opts, new_opts_dict)
    pprint.pprint(opts)
    opts = Namespace(**opts)
    if opts.sub_exp_dir is not None:
        sub_exp_dir = opts.sub_exp_dir
        opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
        create_initial_experiment_dir(opts)
    return opts, previous_train_ckpt


def setup_progressive_steps(opts):
    log_size = int(math.log(opts.stylegan_size, 2))
    num_style_layers = 2 * log_size - 8
    num_deltas = num_style_layers - 1
    if opts.progressive_start is not None:  # If progressive delta training
        opts.progressive_steps = [0]
        next_progressive_step = opts.progressive_start
        for i in range(num_deltas):
            opts.progressive_steps.append(next_progressive_step)
            next_progressive_step += opts.progressive_step_every

    assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
        "Invalid progressive training input"


def is_valid_progressive_steps(opts, num_style_layers):
    return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts):
    if not os.path.exists(opts.exp_dir):
        os.makedirs(opts.exp_dir)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
    for k, v in new_opts.items():
        if k not in ckpt_opts:
            ckpt_opts[k] = v
    if new_opts['update_param_list']:
        for param in new_opts['update_param_list']:
            ckpt_opts[param] = new_opts[param]


if __name__ == '__main__':
    main()