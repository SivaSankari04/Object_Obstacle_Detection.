import cv2
import argparse
from ultralytics import YOLO
import os
from collections import Counter

# Constants (tweak these based on your setup)
FOCAL_LENGTH = 950
KNOWN_WIDTHS = {
    'pen': 0.1, 'cell phone': 0.07, 'person': 0.5, 'notebook': 0.3,
    'bottle': 0.10, 'bat': 1.0, 'ball': 0.2, 'computer': 0.5,
    'bird': 0.3, 'fan': 0.4, 'light': 0.25
}

model = YOLO("yolov8m.pt")

def estimate_distance(pix_w, label):
    if label in KNOWN_WIDTHS:
        return round((KNOWN_WIDTHS[label] * FOCAL_LENGTH) / pix_w, 2)
    return None

def draw_and_count(results, frame, estimate_dist=False):
    counts = Counter()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            counts[label] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            text = label
            if estimate_dist:
                dist = estimate_distance(x2 - x1, label)
                if dist: text += f" ({dist} m)"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame, counts

def process_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)
        frame, counts = draw_and_count(results, frame, estimate_dist=True)
        y = 30
        for lbl, cnt in counts.items():
            cv2.putText(frame, f"{lbl}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            y += 25
        cv2.imshow("Webcam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

def process_image(path):
    img = cv2.imread(path)
    results = model(img)
    img, counts = draw_and_count(results, img)
    y = 30
    for lbl, cnt in counts.items():
        cv2.putText(img, f"{lbl}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y += 25
    cv2.imshow("Image Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(path):
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)
        frame, counts = draw_and_count(results, frame)
        y = 30
        for lbl, cnt in counts.items():
            cv2.putText(frame, f"{lbl}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            y += 25
        cv2.imshow("Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, choices=["webcam", "image", "video"], required=True,
                        help="Specify input source type")
    parser.add_argument("--image_path", type=str, help="Path to input image (if source is 'image')")
    parser.add_argument("--video_path", type=str, help="Path to input video (if source is 'video')")
    args = parser.parse_args()

    if args.source == "webcam":
        process_webcam()
    elif args.source == "image":
        if args.image_path and os.path.isfile(args.image_path):
            process_image(args.image_path)
        else:
            print("Please provide a valid --image_path.")
    elif args.source == "video":
        if args.video_path and os.path.isfile(args.video_path):
            process_video(args.video_path)
        else:
            print("Please provide a valid --video_path.")
