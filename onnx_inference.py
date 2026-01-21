from ultralytics import YOLO
import os

# Create outputs folder if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load the YOLO ONNX model
model_path = "model/yolo11n.onnx"
model = YOLO(model_path)

# Run inference on the image
image_path = "data/image.png"
results = model(image_path)

# Process results and print detections
print("\n" + "="*60)
print("Object Detection Results (ONNX)")
print("="*60)

for i, result in enumerate(results):
    print(f"\nImage {i+1}: {image_path}")
    print(f"Number of detections: {len(result.boxes)}\n")
    
    # Print each detection
    for j, box in enumerate(result.boxes):
        # Get class ID, confidence, and bounding box coordinates
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        class_name = model.names[class_id]
        
        print(f"Detection {j+1}:")
        print(f"  Class ID: {class_id}")
        print(f"  Class Name: {class_name}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Bounding Box: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
        print()

# Save the annotated image
output_path = os.path.join(output_dir, "onnx_result.png")
result_image = results[0].plot()  # Get annotated image
results[0].save(output_path)  # Save to file

print("="*60)
print(f"Annotated image saved to: {output_path}")
print("="*60)

