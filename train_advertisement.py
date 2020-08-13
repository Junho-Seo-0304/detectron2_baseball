from detectron2.data.datasets import register_coco_instances
register_coco_instances("baseball_advertisement", {}, "./advertisement/result.json", "./advertisement/")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("baseball_advertisement",)
cfg.DATASETS.TEST = ()  
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00005
cfg.OUTPUT_DIR = cfg.OUTPUT_DIR+'_advertisement'
cfg.SOLVER.MAX_ITER = (
    10000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
