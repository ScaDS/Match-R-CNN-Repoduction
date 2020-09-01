import cv2
import os

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances

import settings

register_coco_instances('train',
                        {},
                        os.path.join(settings.DATA_DIR,
                                     'processed',
                                     'deepfashion2_coco_train.json'),
                        os.path.join(settings.DATA_DIR,
                                     'raw',
                                     'train',
                                     'image'))
register_coco_instances("validation",
                        {},
                        os.path.join(settings.DATA_DIR,
                                     'processed',
                                     'deepfashion2_coco_validation.json'),
                        os.path.join(settings.DATA_DIR,
                                     'raw',
                                     'validation',
                                     'image'))
register_coco_instances("test",
                        {},
                        os.path.join(settings.DATA_DIR,
                                     'processed',
                                     'deepfashion2_coco_test.json'),
                        os.path.join(settings.DATA_DIR,
                                     'raw',
                                     'test',
                                     'image'))

train_metadata = MetadataCatalog.get('train')
dataset_dicts = DatasetCatalog.get('train')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('train',)
cfg.DATASETS.TEST = ('test',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
evaluator = COCOEvaluator('validation', cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, 'validation')
print(inference_on_dataset(trainer.model, val_loader, evaluator))


