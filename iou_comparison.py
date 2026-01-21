from ultralytics import YOLO
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union_area = area1 + area2 - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # Calculate IoU
    iou = intersection_area / union_area
    return iou

# Load both models
pytorch_model_path = "model/yolo11n.pt"
onnx_model_path = "model/yolo11n.onnx"

print("Loading models...")
pytorch_model = YOLO(pytorch_model_path)
onnx_model = YOLO(onnx_model_path)

# Run inference on the image
image_path = "data/image.png"
print(f"Running inference on {image_path}...\n")

pytorch_results = pytorch_model(image_path)
onnx_results = onnx_model(image_path)

# Extract detections from both models
pytorch_result = pytorch_results[0]
onnx_result = onnx_results[0]

pytorch_boxes = []
onnx_boxes = []

# Extract PyTorch detections
for box in pytorch_result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    bbox = box.xyxy[0].tolist()
    pytorch_boxes.append({
        'class_id': class_id,
        'class_name': pytorch_model.names[class_id],
        'confidence': confidence,
        'bbox': bbox
    })

# Extract ONNX detections
for box in onnx_result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    bbox = box.xyxy[0].tolist()
    onnx_boxes.append({
        'class_id': class_id,
        'class_name': onnx_model.names[class_id],
        'confidence': confidence,
        'bbox': bbox
    })

print("="*80)
print("IoU Comparison Results")
print("="*80)
print(f"PyTorch model detections: {len(pytorch_boxes)}")
print(f"ONNX model detections: {len(onnx_boxes)}")
print()

# Compute IoU for matching detections
# For each PyTorch detection, find the best matching ONNX detection of the same class
comparison_results = []

for i, pt_box in enumerate(pytorch_boxes):
    best_iou = 0.0
    best_match_idx = None
    best_match = None
    
    # Find all ONNX detections with the same class
    matching_onnx_boxes = [j for j, onnx_box in enumerate(onnx_boxes) 
                          if onnx_box['class_id'] == pt_box['class_id']]
    
    # Calculate IoU with all matching detections
    for j in matching_onnx_boxes:
        onnx_box = onnx_boxes[j]
        iou = calculate_iou(pt_box['bbox'], onnx_box['bbox'])
        
        if iou > best_iou:
            best_iou = iou
            best_match_idx = j
            best_match = onnx_box
    
    comparison_results.append({
        'pytorch_idx': i,
        'pytorch_box': pt_box,
        'best_match': best_match,
        'best_match_idx': best_match_idx,
        'best_iou': best_iou
    })

# Print results in a readable format
print("Best IoU per Object (PyTorch → ONNX):")
print("-" * 80)

for result in comparison_results:
    pt_box = result['pytorch_box']
    best_iou = result['best_iou']
    best_match = result['best_match']
    
    print(f"\nObject {result['pytorch_idx'] + 1}:")
    print(f"  Class: {pt_box['class_name']} (ID: {pt_box['class_id']})")
    print(f"  PyTorch Detection:")
    print(f"    Confidence: {pt_box['confidence']:.4f}")
    print(f"    BBox: [{pt_box['bbox'][0]:.2f}, {pt_box['bbox'][1]:.2f}, "
          f"{pt_box['bbox'][2]:.2f}, {pt_box['bbox'][3]:.2f}]")
    
    if best_match:
        print(f"  Best ONNX Match:")
        print(f"    Confidence: {best_match['confidence']:.4f}")
        print(f"    BBox: [{best_match['bbox'][0]:.2f}, {best_match['bbox'][1]:.2f}, "
              f"{best_match['bbox'][2]:.2f}, {best_match['bbox'][3]:.2f}]")
        print(f"  Best IoU: {best_iou:.4f}")
    else:
        print(f"  Best ONNX Match: No matching class found")
        print(f"  Best IoU: N/A")

print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

iou_values = [r['best_iou'] for r in comparison_results if r['best_match'] is not None]
if iou_values:
    print(f"Average IoU: {np.mean(iou_values):.4f}")
    print(f"Minimum IoU: {np.min(iou_values):.4f}")
    print(f"Maximum IoU: {np.max(iou_values):.4f}")
    print(f"Objects with IoU > 0.5: {sum(1 for iou in iou_values if iou > 0.5)}/{len(iou_values)}")
    print(f"Objects with IoU > 0.7: {sum(1 for iou in iou_values if iou > 0.7)}/{len(iou_values)}")
else:
    print("No matching detections found between models.")

print("="*80)

