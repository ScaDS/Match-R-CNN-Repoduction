import cv2
import os
import random

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

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
                                     'image',
                                     'validation'))

train_metadata = MetadataCatalog.get('train')
dataset_dicts = DatasetCatalog.get('train')

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d['file_name'])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('image', out.get_image()[:, :, ::-1])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('train',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ('validation', )
predictor = DefaultPredictor(cfg)

img = cv2.imread('demo/input.jpg')
outputs = predictor(img)

for d in random.sample(dataset_dicts, 1):
    im = cv2.imread(d['file_name'])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=train_metadata,
                   scale=0.8)
    out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    cv2.imwrite('demo/output_retrained.jpg', out.get_image()[:, :, ::-1])
