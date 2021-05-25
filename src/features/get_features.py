import os
# import cv2
# from PIL import Image

# import numpy
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import ImageList


# def get_box_features(img):
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     cfg.SOLVER.IMS_PER_BATCH = 1
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
#     model = build_model(cfg)
#     DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
#     model.eval()
#
#     # image = cv2.imread(img)
#     pil_image = img.convert('RGB')
#     open_cv_image = numpy.array(pil_image)
#     image = open_cv_image[:, :, ::-1].copy()
#
#     height, width = image.shape[:2]
#     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#     inputs = [{"image": image, "height": height, "width": width}]
#     with torch.no_grad():
#         images = model.preprocess_image(inputs)
#         features = model.backbone(images.tensor)  # set of cnn features
#         proposals, _ = model.proposal_generator(images, features, None)  # RPN
#
#         features_ = [features[f] for f in model.roi_heads.box_in_features]
#         box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
#         box_features = model.roi_heads.box_head(box_features)
#         predictions = model.roi_heads.box_predictor(box_features)
#         pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
#         # pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
#
#         # output boxes, masks, scores, etc
#         # pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
#         # features of the proposed boxes
#         feats = box_features[pred_inds]
#     # return feats, pred_instances
#     return feats


def get_features(images_list):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    with torch.no_grad():
        images = ImageList.from_tensors(images_list, size_divisibility=32).to('cuda')
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)

        feats = box_features[pred_inds]
    return feats
