import os
from detectron2.structures import BoxMode
import settings
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train",
                        {},
                        os.path.join(settings.DATA_DIR,
                                     'processed',
                                     'deepfashion2_train_json'),
                        os.path.join(settings.DATA_DIR,
                                     'train',
                                     'image'))
register_coco_instances("my_dataset_val",
                        {},
                        os.path.join(settings.DATA_DIR,
                                     'processed',
                                     'deepfashion2_validation_json'),
                        os.path.join(settings.DATA_DIR,
                                     'validation',
                                     'image'))
