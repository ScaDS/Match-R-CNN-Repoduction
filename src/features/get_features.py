import argparse
import os
import pickle
from os import listdir
from pathlib import Path

import cv2
import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from tqdm import tqdm


def get_features(images_list, model):

    with torch.no_grad():
        # images = ImageList.from_tensors(images_list, size_divisibility=32).to('cuda')
        images = model.preprocess_image(images_list)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features)
        instances, _ = model.roi_heads(images, features, proposals)
        mask_features = [features[f] for f in model.roi_heads.in_features]
        mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])

    return mask_features


def make_features_list(image_dir, model):

    image_list = listdir(image_dir)

    feature_file = open(os.path.join('data', 'results', 'features_train.txt'), 'a')

    with torch.no_grad():
        for img in tqdm(image_list):
            img_path = os.path.join(image_dir, img)
            image = cv2.imread(img_path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]

            feature = get_features(inputs, model)
            feat_tuple = (Path(img).stem, feature)

            feature_file.write(str(feat_tuple))

        feature_file.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        help='image path',
                        type=str,
                        default=os.path.join('data', 'raw', 'train', 'image'))
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join('output', 'model_final.pth')
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    make_features_list(args.path, model)


if __name__ == '__main__':
    main()
