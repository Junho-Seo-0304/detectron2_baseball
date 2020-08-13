from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import cv2

path_dir = './test_dataset'
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
baseball_metadata = MetadataCatalog.get("baseball_setup_faster")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR+'_setup_faster', "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.DATASETS.TEST = ("baseball_setup_faster",)
predictor = DefaultPredictor(cfg)



count = 0
error_count = 0
miss_count = 0

for root, dirs, files in os.walk(path_dir):
    for file in files:
        s = os.path.splitext(file)
        if s[1] == '.jpg':
            filepath = os.path.join(root, file)
            print(filepath)
            im = cv2.imread(filepath)
            outputs = predictor(im)
            exist = False
            for classes, scores in zip(outputs["instances"].pred_classes,outputs["instances"].scores):
                if classes == 2:
                    exist = True
            if exist:
                count=count+1
                print(count)
                if 'SetupX' in filepath:
                    error_count = error_count+1
                    print(error_count)
            else:
                if 'SetupX' not in filepath:
                    miss_count=miss_count+1
print('Total : ',count)
print('error : ',error_count)
print('miss : ',miss_count)