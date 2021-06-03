from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

def infer(path,weights, threshold_score):

    im = cv2.imread(path)

    # Create config
    cfg = get_cfg()
    if weights == "faster_rcnn_R_101_FPN_3x":
        cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    elif weights == "mask_rcnn_R_101_FPN_3x":
        cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"

        
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_score  # set threshold for this model
    cfg.MODEL.DEVICE = "cpu"

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Make prediction
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('sample.jpg', v.get_image()[:, :, ::-1])

if __name__ == "__main__":
    infer("./input.jpg", "mask_rcnn_R_101_FPN_3x", 0.5)
    
