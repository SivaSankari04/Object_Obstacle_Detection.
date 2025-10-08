# Real-time object and obstacle detection using YOLOv8. 
Supports images, videos, and live webcam input. 
A basic distance estimation is included for known objects based on their real-world width. 

NOTE:Detection may not always be 100% accurate; some misclassifications can occur. 

## Features 
-Detect multiple objects in images, videos, or live webcam.

-Count objects and display bounding boxes with labels. 

-Estimate approximate distance for known objects (in meters). 

## Installation & Usage Clone the repository: 
git clone https://github.com/SivaSankari04/Object_Obstacle_Detection.git

2.Install required Python packages: pip install opencv-python numpy ultralytics and YOLO model files (yolov8m.pt)

3.Commands to run (preferably in VScode)

a)To Detect objects in an image:

python Detection.py --source image --image_path "C:\Obstacle Detection\Samples\image1.jpg"

python Detection.py --source image --image_path "C:\Obstacle Detection\Samples\image2.jpeg"

b)To Detect objects in a video: 

python Detection.py --source video --video_path "C:\Obstacle Detection\Samples\Samples\cars.mp4" 

c)To Detect objects via webcam: python Detection.py --source webcam 

Press q or Ctrl+C to quit the webcam/video window.


Output ScreenShot:
<img width="1912" height="1011" alt="Sample Output" src="https://github.com/user-attachments/assets/a2472af2-60c8-4183-b575-3a974cccc0cd" />
