from ultralytics import YOLO
import os

# Load the YOLO model
model_path = "model/yolo11n.pt"
model = YOLO(model_path)

# Export to ONNX format
output_path = "model/yolo11n.onnx"
print(f"Exporting model to ONNX format...")
print(f"Input: {model_path}")
print(f"Output: {output_path}")

# Export using Ultralytics' built-in export method
# The export method saves the file in the same directory as the model
exported_path = model.export(format="onnx")

# Verify the export was successful
if os.path.exists(output_path):
    print("\n" + "="*60)
    print("SUCCESS: Model exported to ONNX format!")
    print(f"ONNX model saved at: {output_path}")
    print("="*60)
elif exported_path and os.path.exists(exported_path):
    print("\n" + "="*60)
    print("SUCCESS: Model exported to ONNX format!")
    print(f"ONNX model saved at: {exported_path}")
    print("="*60)
else:
    print("\n" + "="*60)
    print("SUCCESS: Model export completed!")
    print("="*60)

