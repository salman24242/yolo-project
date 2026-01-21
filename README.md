**YOLO Object Detection Project**
A computer vision project using YOLOv11 for real-time object detection on images and videos.

Features:
1) Image object detection with bounding boxes
2) Video object detection with frame-by-frame analysis
3) Support for 80 COCO dataset classes (person, car, dog, etc.)
4) Automated output saving with annotations

**Requirements**

Python 3.8+
PyTorch
Ultralytics YOLO
OpenCV

**Installation**

**Clone this repository:**

git clone https://github.com/salman24242/yolo_project.git
cd yolo_project

**Create a virtual environment:**

python -m venv venv          #Create virual environment
venv\Scripts\activate        #Activate the virtual environment

**Install dependencies:**

pip install -r requirements.txt

**Download YOLO model weights:**
You can download it from the Ultralytics 


**Usage**
**Image Detection**
python pytorch_inference.py
Place your image in data/image.png and run the script. Output will be saved to outputs/output_image.png.

**Video Detection**
python yolo_video_detection.py
Place your video in data/video.mp4 and run the script. Output will be saved to outputs/output_video.mp4.


**Detectable Objects**
The model can detect 80 classes including:

People, vehicles (car, truck, bus, bicycle, motorcycle)
Animals (dog, cat, bird, horse, etc.)
Common objects (chair, laptop, phone, etc.)

See full list in COCO dataset classes.
**Results**
Detection results include:

Class name and ID
Confidence score
Bounding box coordinates
Annotated output images/videos

**Acknowledgments**

Ultralytics YOLO
COCO Dataset

**Contact**
Salman - salman.ta12312@gmail.com
**Project Link:** https://github.com/salman24242/yolo_project
