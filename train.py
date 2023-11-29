import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import copy


from runners.diffusionmat import Diffusion










def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--delta_config', type=str, required=True, help='Path to the config file')

    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=10, help='Total sampling steps')

    parser.add_argument('--warmup_step', type=int, default=5000, help='warmup steps for training')
    parser.add_argument('--total_step', type=int, default=1000000, help='total steps for training')

    parser.add_argument('--resume_training', action='store_true')

    parser.add_argument('--w_inv', type=float, default=2, help='total steps for training')
    parser.add_argument('--w_com', type=float, default=1, help='total steps for training')
    parser.add_argument('--w_alpha', type=float, default=1, help='total steps for training')
    

    parser.add_argument('--t', type=int, default=500, help='Sampling noise scale')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    with open(os.path.join('configs', args.delta_config), 'r') as f:
        config = yaml.safe_load(f)
    delta_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, 'image')
    args.ckpt_folder = os.path.join(args.exp, 'checkpoints')
    args.log_folder = os.path.join(args.exp, 'logs')

    
    if not os.path.exists(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:

            overwrite = False

        if overwrite:
            shutil.rmtree(args.ckpt_folder)
            os.makedirs(args.ckpt_folder)
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
            shutil.rmtree(args.log_folder)
            os.makedirs(args.log_folder)
        # else:
        #     print("Output image folder exists. Program halted.")
        #     sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, delta_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config, delta_config = parse_args_and_config()
    print (config)
    print (delta_config)

    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)

    try:
        runner = Diffusion(args, config, delta_config)
        runner.image_matting_train()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
