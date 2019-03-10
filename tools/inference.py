# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import cv2
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.inference import get_final_preds
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def main():
    args = parse_args()
    update_config(cfg, args)

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    MODEL_TXT_FILE = '../output/model.txt'
    os.makedirs(os.path.dirname(MODEL_TXT_FILE), exist_ok=True)
    with open(MODEL_TXT_FILE, 'wt') as f:
        print(model, file=f)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(valid_loader):

            output = model(input)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()

            filenames = meta['image']

            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), c, s)

            for p, img, pred, filename in zip(output, input, preds, filenames):
                # img = unnormalize(img)
                # img_nhwc = np.transpose(img.numpy(), [1, 2, 0])
                # keypoints = CocoKeyPoints(pred)
                # img_with_keypoints = draw_keypoints(img_nhwc, keypoints)

                img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                np.transpose(img)
                keypoints = CocoKeyPoints(pred)
                img_with_keypoints = draw_keypoints(img, keypoints)

                plt.imshow(img_with_keypoints)
                plt.show()


def draw_keypoints(image, keypoints):
    lines, colors = keypoints.lines()
    for line, color in zip(lines, colors):
        cv2.line(image, line[0], line[1], color, thickness=3)

    return image


class CocoKeyPoints:
    def __init__(self, keypoints):
        keypoints = keypoints.astype(int)
        keypoints = [tuple(k.tolist()) for k in keypoints]
        self.keypoints = dict(
            nose=keypoints[0],
            left_eye=keypoints[1],
            right_eye=keypoints[2],
            left_ear=keypoints[3],
            right_ear=keypoints[4],
            left_shoulder=keypoints[5],
            right_shoulder=keypoints[6],
            left_elbow=keypoints[7],
            right_elbow=keypoints[8],
            left_wrist=keypoints[9],
            right_wrist=keypoints[10],
            left_hip=keypoints[11],
            right_hip=keypoints[12],
            left_knee=keypoints[13],
            right_knee=keypoints[14],
            left_ankle=keypoints[15],
            right_ankle=keypoints[16],
        )

    def lines(self):
        _lines = [
            (self.keypoints['nose'], self.keypoints['left_eye']),
            (self.keypoints['nose'], self.keypoints['right_eye']),
            (self.keypoints['left_eye'], self.keypoints['left_ear']),
            (self.keypoints['right_eye'], self.keypoints['right_ear']),
            (self.keypoints['left_shoulder'], self.keypoints['right_shoulder']),
            (self.keypoints['left_shoulder'], self.keypoints['left_elbow']),
            (self.keypoints['right_shoulder'], self.keypoints['right_elbow']),
            (self.keypoints['left_wrist'], self.keypoints['left_elbow']),
            (self.keypoints['right_wrist'], self.keypoints['right_elbow']),
            (self.keypoints['left_shoulder'], self.keypoints['left_hip']),
            (self.keypoints['right_shoulder'], self.keypoints['right_hip']),
            (self.keypoints['left_knee'], self.keypoints['left_hip']),
            (self.keypoints['right_knee'], self.keypoints['right_hip']),
            (self.keypoints['left_knee'], self.keypoints['left_ankle']),
            (self.keypoints['right_knee'], self.keypoints['right_ankle']),
        ]
        _colors = ((200, 0, 0),) * len(_lines)
        return _lines, _colors


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
                + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
                shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))


if __name__ == '__main__':
    main()
