import os
from inference import infer

def main(toolbox, path, weights, threshold_score):

    if toolbox == "detectron2":
        infer(path, weights, threshold_score)

    if toolbox == "yolov5":
        os.system("python yolov5/detect.py --source {} --weights {} --conf {}".format(path, weights, threshold_score))

if __name__ == "__main__":
    # main("detectron2", "./input.jpg", "mask_rcnn_R_101_FPN_3x", 0.5)
    main("yolov5", "input.jpg", "yolov5s.pt", 0.25)