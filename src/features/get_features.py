import os
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import ImageList


def get_features(images_list):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join('output', 'model_final.pth')
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    with torch.no_grad():
        images = ImageList.from_tensors(images_list, size_divisibility=32).to('cuda')

        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features)
        instances, _ = model.roi_heads(images, features, proposals)
        mask_features = [features[f] for f in model.roi_heads.in_features]
        mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])

    return mask_features
