import io
import json
import base64
import uuid
import logging
import sys
from typing import List
from PIL import Image
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import os
import cv2
from pydantic import BaseModel
from inference import infer



app = FastAPI()

class Data(BaseModel):
    file: str
    weights: str
    threshold_score: float

@app.post("/detectron2")
async def main(data: Data):
    infer(data.file, data.weights, data.threshold_score)
    # img = cv2.imread("sample.jpg")
    # res, im_png = cv2.imencode(".png", img)
    # return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/yolov5")
def infer_yolov5(data: Data):
    os.system("rm -rf runs/detect/exp")
    os.system("python yolov5/detect.py --source {} --weights {} --conf {}".format(data.file, data.weights, data.threshold_score))
#     filename = os.path.basename(data.path)
#     img = cv2.imread("runs/detect/exp/{}".format(filename))
#     res, im_png = cv2.imencode(".png", img)
#     return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

