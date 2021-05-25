import logging
from collections import OrderedDict

from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks, launch
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import os

# import settings
from detectron2.modeling import GeneralizedRCNNWithTTA


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


register_coco_instances(name='train',
                        metadata={},
                        # json_file=os.path.join(settings.DATA_DIR, 'processed', 'deepfashion2_coco_train.json'),
                        json_file=os.path.join('data', 'processed', 'deepfashion2_coco_train.json'),
                        image_root=os.path.join('data', 'raw', 'train', 'image'))
register_coco_instances(name='validation',
                        metadata={},
                        json_file=os.path.join('data', 'processed', 'deepfashion2_coco_validation.json'),
                        image_root=os.path.join('data', 'raw', 'validation', 'image'))
register_coco_instances(name='test',
                        metadata={},
                        json_file=os.path.join('data', 'processed', 'deepfashion2_coco_test.json'),
                        image_root=os.path.join('data', 'raw', 'test', 'image'))
# train_metadata = MetadataCatalog.get('train')
train_dataset_dicts = DatasetCatalog.get('train')
validation_dataset_dicts = DatasetCatalog.get('validation')
test_dataset_dicts = DatasetCatalog = DatasetCatalog.get('test')


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    # cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.TEST = ('validation',)
    cfg.TEST.EVAL_PERIOD = 21622
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 259464
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (172976,)
    cfg.SOLVER.WEIGHT_DECAY = 0.00001
    return cfg


def main():
    cfg = setup()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    evaluator = COCOEvaluator("validation", cfg, True, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "validation")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


# def main():
#     cfg = setup()
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     trainer = Trainer(cfg)
#     trainer.resume_or_load(resume=True)
#     if cfg.TEST.AUG.ENABLED:
#         trainer.register_hooks(
#             [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
#         )
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     return trainer.train()


if __name__ == '__main__':
    main()


# if __name__ == "__main__":
#     launch(
#         main,
#         2,
#         num_machines=1,
#         machine_rank=0,
#         dist_url='auto',
#         args=(),
#     )
