from detectron2.data.datasets import register_coco_instances
register_coco_instances("baseball_setup", {}, "./baseball_2019_setup_without_0/result.json", "./baseball_2019_setup_without_0/")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
print(cfg)
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("baseball_setup",)
cfg.DATASETS.TEST = ()  
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00005
cfg.OUTPUT_DIR = cfg.OUTPUT_DIR+'_setup'
cfg.SOLVER.MAX_ITER = (
    10000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.RETINANET.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.RETINANET.NUM_CLASSES = 3  # 3 classes 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
