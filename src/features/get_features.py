import argparse
import os
from os import listdir
from pathlib import Path

import cv2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model

import torch
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


def get_box_features(images_list, model):
    with torch.no_grad():
        images = model.preprocess_image(images_list)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, images_list, images.image_sizes)  # scale box to orig size
        # features of the proposed boxes
        feats = box_features[pred_inds]

    return feats, pred_instances


def make_features_list(image_dir, model):

    image_list = listdir(image_dir)

    feature_file = open(os.path.join('data', 'results', 'box_features_train.txt'), 'a')

    with torch.no_grad():
        for img in tqdm(image_list):
            img_path = os.path.join(image_dir, img)
            image = cv2.imread(img_path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]

            # feature = get_features(inputs, model)
            feature = get_box_features(inputs, model)
            # feature_file.write('id:')
            # feature_file.write(Path(img).stem)
            # feature_file.write(str(list(feature[0])))

            for feat in feature[0]:
                feature_file.write(Path(img).stem + ' ' + str(feat.tolist()) + '\n')

        feature_file.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        help='image path',
                        type=str,
                        default=os.path.join('data', 'raw', 'train', 'image'))
                        # default='test_feat')
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join('output', 'model_final.pth')
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.INPUT.MIN_SIZE_TEST = 800

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    make_features_list(args.path, model)


if __name__ == '__main__':
    main()
