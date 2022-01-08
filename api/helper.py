import os
import cv2
import numpy as np


def get_image_paths(folder, extension):
  image_paths = []
  for x in os.listdir(folder):
    x_path = os.path.join(folder, x)
    if os.path.splitext(x_path)[1] in extension:
      image_paths.append(x_path)
  return image_paths


def load_data_and_labels(folder, label):
  images = []
  labels = []
  image_paths = get_image_paths(folder, ['.jpg', '.png', '.jpeg'])
  for image_path in image_paths:
    im = cv2.imread(image_path, cv2.IMREAD_COLOR)
    images.append(im)
    labels.append(label)
  return images, labels

def get_svm_detector_for_hog(model_path, hog):
  svm = cv2.ml.SVM_load(model_path)
  sv = svm.getSupportVectors()
  rho,_ ,_ = svm.getDecisionFunction(0)
  svm_detector = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
  svm_detector[:-1] = -sv[:]
  svm_detector[-1] = rho
  return svm_detector

def detect(img_batch, hog, encode_image, color, winStride, hitThreshold):
    img_str_list = [] 
    bbox_list = []
    for im in img_batch:
        finalHeight = 840.0
        scale = finalHeight / im.shape[0]
        im = cv2.resize(im, None, fx=scale, fy=scale)
        bboxes, weights = hog.detectMultiScale(im, winStride = winStride,
                                        padding=(32, 32),scale=1.05, 
                                        finalThreshold=2,hitThreshold = hitThreshold)
        # print(weights)
        for bbox in bboxes:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(im, (x1, y1), (x2, y2), 
                            color, thickness=3, 
                            lineType=cv2.LINE_AA)

        img_str_list.append(encode_image(im))
        bbox_list.append(bboxes)
    
    return img_str_list, bbox_list
