from ultralytics import YOLO
import cv2

def run_test():
    print("--- Team Coconut: Initializing YOLO Test ---")
    
    # 1. Load a small pretrained YOLOv8 model (downloads automatically)
    model = YOLO('yolov8n.pt') 

    # 2. Run detection on a sample image from the web
    source = 'https://ultralytics.com/images/bus.jpg'
    results = model.predict(source, save=True, imgsz=640)

    # 3. Print results to console
    for result in results:
        print(f"Detected {len(result.boxes)} objects in the image.")
        for box in result.boxes:
            print(f"Class: {model.names[int(box.cls)]} | Confidence: {box.conf.item():.2f}")

    print("\nSuccess! Results saved to the 'runs/detect/predict' folder.")

if __name__ == "__main__":
    run_test()