import cv2
import numpy as np
import torch
import base64
import random
from api.helper import detect, get_svm_detector_for_hog
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from typing import List
# import settings

app = FastAPI()
templates = Jinja2Templates(directory = 'api/templates')
model_selection_options = ['yolov5s','pedestrian', 'pedestrian_default']
model_dict = {'yolov5s' : '../model/yolo5s.pt', 'pedestrian': '../model/pedestrian.yml', 'pedestrian_default':'../model/pedestrian2.yml'} #set up model cache
colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting


@app.get("/")
def home(request: Request):
	return templates.TemplateResponse('home.html', {
			"request": request,
			"model_selection_options": model_selection_options,
		})


@app.post("/")
async def detect_via_web_form(request: Request,
							file_list: List[UploadFile] = File(...), 
							model_name: str = Form(...),
							img_size: int = Form(800)):

	img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)
				for file in file_list]

	if model_name == 'yolov5s':
		model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
		results = model_dict[model_name](img_batch.copy(), size = img_size)
		json_results = results_to_json(results,model_dict[model_name])
		img_str_list = []
		for img, bbox_list in zip(img_batch, json_results):
			for bbox in bbox_list:
				label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
				plot_one_box(bbox['bbox'], img, label=label, 
						color=colors[int(bbox['class'])], line_thickness=3)
			img_str_list.append(base64EncodeImage(img))
		encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

		return templates.TemplateResponse('show_results.html', {
				'request': request,
				'bbox_image_data_zipped': zip(img_str_list,json_results), 
			})

	elif model_name == 'pedestrian':
		window = (64, 128)
		block = (16, 16)
		block_stride = (8, 8)
		cell = (8, 8)
		n_bins = 9
		derivative_aperature = 1
		sigma = -1
		norm_type = 0
		threshold = 0.2
		gamma_correction = True
		nlevels = 64
		gradient = False

		hog = cv2.HOGDescriptor(window, block, block_stride,
							cell, n_bins,derivative_aperature,
							sigma, norm_type, threshold, 
							gamma_correction, nlevels,gradient)
				
		svm_detector = get_svm_detector_for_hog('model/pedestrian2.yml', hog)
		hog.setSVMDetector(svm_detector)
		img_str_list = []
		green = (0, 255, 0)
		hitThreshold = 1.0
		winStride = (4, 4)
		img_str_list, json_results = detect(img_batch, hog, base64EncodeImage, green, winStride, hitThreshold)

		return templates.TemplateResponse('show_results.html', {
		'request': request,
		'bbox_image_data_zipped': zip(img_str_list,json_results),
	    })

	else: 
		window = (64, 128)
		block = (16, 16)
		block_stride = (8, 8)
		cell = (8, 8)
		n_bins = 9
		derivative_aperature = 1
		sigma = -1
		norm_type = 0
		threshold = 0.2
		gamma_correction = True
		nlevels = 64
		gradient = False

		hog_lib = cv2.HOGDescriptor(window, block, block_stride,
							cell, n_bins,derivative_aperature,
							sigma, norm_type, threshold, 
							gamma_correction, nlevels,gradient)
		svm_detector_default = cv2.HOGDescriptor_getDefaultPeopleDetector()
		hog_lib.setSVMDetector(svm_detector_default)
		img_str_list = []
		red = (0, 0, 255)
		hitThreshold = 0
		winStride = (8, 8)
		img_str_list, json_results = detect(img_batch, hog_lib, base64EncodeImage, red, winStride, hitThreshold)

		return templates.TemplateResponse('show_results.html', {
		'request': request,
		'bbox_image_data_zipped': zip(img_str_list,json_results),
	    })


def results_to_json(results, model):
	''' Converts yolo model output to json (list of list of dicts)'''
	return [
				[
					{
					"class": int(pred[5]),
					"class_name": model.model.names[int(pred[5])],
					"bbox": [int(x) for x in pred[:4].tolist()],
					"confidence": float(pred[4]),
					}
				for pred in result
				]
			for result in results.xyxy
			]


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def base64EncodeImage(img):
	''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
	_, im_arr = cv2.imencode('.jpg', img)
	im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')
	return im_b64

if __name__ == '__main__':
	import uvicorn
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', default = 'localhost')
	parser.add_argument('--port', default = 8000)
	# parser.add_argument('--precache-models', action='store_true', help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
	opt = parser.parse_args()

	# if opt.precache_models:
	# 	model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True) 
	# 					for model_name in model_selection_options}
	
	app_str = 'server:app' 
	uvicorn.run(app_str, host= opt.host, port=opt.port, reload=True)
