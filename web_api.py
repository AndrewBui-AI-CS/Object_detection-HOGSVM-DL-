import os
import io
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import io
import uvicorn
import requests
import numpy as np
import nest_asyncio
from enum import Enum
from IPython.display import Image, display
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"

dir_name = "images_with_boxes"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name = "images_uploaded"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name = "images_predicted"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

app = FastAPI(title='Human detection system')
templates = Jinja2Templates(directory=".")
@app.get("/")

@app.post("/predict") 

def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def detect_and_draw_box(filename, model="yolov3-tiny", confidence=0.5):
    img_filepath = f'../images/{filename}'
    img = cv2.imread(img_filepath)
    bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)
    print(f"========================\nImage processed: {filename}\n")
    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")
    output_image = draw_bbox(img, bbox, label, conf)
    cv2.imwrite(f'images_with_boxes/{filename}', output_image)
    display(Image(f'images_with_boxes/{filename}'))


def prediction(model: Model, file: UploadFile = File(...)):
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    bbox, label, conf = cv.detect_common_objects(image, model=model)
    output_image = draw_bbox(image, bbox, label, conf)
    cv2.imwrite(f'images_uploaded/{filename}', output_image)
    file_image = open(f'images_uploaded/{filename}', mode="rb")
    return StreamingResponse(file_image, media_type="image/jpeg")


def response_from_server(url, image_file, verbose=True):
    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response


def display_image_from_response(response):
    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "image_with_objects.jpeg"
    cv2.imwrite(f'images_predicted/{filename}', image)
    display(Image(f'images_predicted/{filename}'))


nest_asyncio.apply()
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"  
uvicorn.run(app, host=host, port=8000)

base_url = 'http://localhost:8000'
endpoint = '/predict'
model = 'yolov3-tiny'
url_with_endpoint_no_params = base_url + endpoint
full_url = url_with_endpoint_no_params + "?model=" + model
