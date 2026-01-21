from ultralytics import YOLO
import cv2
import os
import urllib.request

# Create outputs folder if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# Function to download video from URL
"""
def download_video(url, save_path):
    Download video from URL to local path
    print(f"Downloading video from {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Video downloaded successfully to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False
"""

# Load the YOLO model
model_path = "model/yolo11n.pt"
model = YOLO(model_path)

# Option 1: Download video from internet
#video_url = "https://www.pexels.com/download/video/3015494/"  # Example URL
#downloaded_video_path = "data/downloaded_video.mp4"

# Uncomment the line below to download video
# download_video(video_url, downloaded_video_path)

# Option 2: Use local video file
# For now, let's assume you'll place a video in the data folder
video_path = "data/video.mp4"  # Change this to your video path

# Check if video exists
if not os.path.exists(video_path):
    print(f"Video not found at {video_path}")
    print("Please either:")
    print("1. Place a video file at 'data/video.mp4'")
    print("2. Uncomment the download_video line and provide a valid URL")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n" + "=" * 60)
print("Video Detection Started")
print("=" * 60)
print(f"Video: {video_path}")
print(f"Resolution: {frame_width}x{frame_height}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print("=" * 60 + "\n")

# Define output video path
output_video_path = os.path.join(output_dir, "output_video.mp4")

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
total_detections = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Run YOLO inference on the frame
        results = model(frame, verbose=False)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Count detections in this frame
        num_detections = len(results[0].boxes)
        total_detections += num_detections

        # Write the annotated frame to output video
        out.write(annotated_frame)

        # Print progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: Frame {frame_count}/{total_frames} ({progress:.1f}%) - Detections: {num_detections}")

        # Optional: Display the frame in real-time (remove if running headless)
        # cv2.imshow('YOLO Detection', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("\nProcessing interrupted by user")

finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Video Detection Complete")
print("=" * 60)
print(f"Total frames processed: {frame_count}")
print(f"Total detections: {total_detections}")
print(f"Average detections per frame: {total_detections / frame_count:.2f}")
print(f"Output video saved to: {output_video_path}")
print("=" * 60)