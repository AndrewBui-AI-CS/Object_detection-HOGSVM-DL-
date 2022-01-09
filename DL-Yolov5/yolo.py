import yolov5
import torch
from PIL import Image
# how to use
# download checkpoints : https://drive.google.com/file/d/1-NWaANYIB_GskOvTGkuf12lc0DIwjXND/view?usp=sharing
# install yolov5: pip install yolov5


def yolo_detect(image_path):
    # pip install yolov5
    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = yolov5.load('last.pt').to(device)

    # set image, you can use api call in that
    # img = 'DL-Yolov5/results/zidane.jpg'
    img = image_path

    # perform inference
    results = model(img)

    # inference with larger input size
    results = model(img, size=1280)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, x2, y1, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # show detection bounding boxes on image
    results.show()

    # save results into "results/" folder, use result
    results.save(save_dir='results/')