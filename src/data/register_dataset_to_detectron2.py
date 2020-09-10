import os

import cv2
import torch

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils import comm
from detectron2.utils.visualizer import ColorMode, Visualizer

import settings


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)


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
cfg.DATASETS.TEST = ('validation',)
cfg.DATASETS.VAL = ('validation',)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = (10000,)
cfg.SOLVER.MAX_ITER = 100000
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_METHOD = 'linear'
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
# cfg.TEST.EVAL_PERIOD = 100

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])
# swap the order of PeriodicWriter and ValidationLoss
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

trainer.resume_or_load(resume=False)
trainer.train()
evaluator = COCOEvaluator('validation', cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, 'validation')
print(inference_on_dataset(trainer.model, val_loader, evaluator))
#
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# predictor = DefaultPredictor(cfg)
#
#
# for d in torch.random.sample(dataset_dicts, 3):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=train_metadata,
#                    scale=0.5,
#                    instance_mode=ColorMode.IMAGE_BW)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
