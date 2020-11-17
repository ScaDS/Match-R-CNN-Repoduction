import random
import re

import cv2

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import os

from detectron2.utils.visualizer import Visualizer, ColorMode

import settings

register_coco_instances(name='train',
                        metadata={},
                        json_file=os.path.join(settings.DATA_DIR, 'processed', 'deepfashion2_coco_train.json'),
                        image_root=os.path.join(settings.DATA_DIR, 'raw', 'train', 'image'))
register_coco_instances(name="validation",
                        metadata={},
                        json_file=os.path.join(settings.DATA_DIR, 'processed', 'deepfashion2_coco_validation.json'),
                        image_root=os.path.join(settings.DATA_DIR, 'raw', 'validation', 'image'))
train_metadata = MetadataCatalog.get('train')
dataset_dicts = DatasetCatalog.get('train')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('train',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 200000

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# # cfg.DATASETS.TEST = ('validation',)
# predictor = DefaultPredictor(cfg)
#
# dataset_dicts_val = DatasetCatalog.get('validation')
# for d in random.sample(dataset_dicts_val, 3):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(img_rgb=im[:, :, ::-1],
#                    metadata=train_metadata,
#                    scale=1.0,
#                    instance_mode=ColorMode.IMAGE_BW)
#     result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     result_image = result.get_image()[:, :, ::-1]
#     out_file_name = re.search(r"(.*)\.", d["file_name"]).group(0)[:-1]
#     out_file_name += "_processed.png"
#     cv2.imwrite(out_file_name, result_image)
#
# evaluator = COCOEvaluator("validation", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "validation")
# print(inference_on_dataset(trainer.model, val_loader, evaluator))
