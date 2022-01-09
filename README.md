# Human detection with Hog-SVM and Yolov5
Detecting pedestrian in images with supports from Computer Vision technique. 
## Description
In this project, we did these three main things: 
-  Introduce a traditional object detection method with Hog descriptor and SVM classifier
-  Try another approach for the problem by including deep learning model Yolov5
-  Build a small web API for visualizing the result of the two approaches. 
## Installation
From the github repo: 
```bash
git clone git@github.com:buiviethoang/Object_detection-HOGSVM-DL-.git
cd Object_detection-HOGSVM-DL-
```

Install all packages needed: 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirement file.
```bash
pip install -r requirement.txt
```

Unzip file from model folder:
```bash
unzip model/pedestrian2.zip -d model/
unzip model/yolov5.zip -d model/
```

How to run: 
From terminal
```python
python3 api/server.py
```

Then on port 8000 in your browser, there will be a quick demo of our project. Let's enjoy :))
```bash
localhost:8000
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors
* **Bùi Việt Hoàng**
* **Nguyễn Thịnh Vượng**
* **Nguyễn Trung Đức**
* **Đinh Công Minh**

## Acknowledgements
I'll update later

## License
[MIT](https://choosealicense.com/licenses/mit/)
