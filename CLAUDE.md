# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YOLO-based agricultural computer vision project for object detection in pastoral/livestock images. The project uses YOLOv8 (via the Ultralytics library) to detect objects in images such as cows in pasture scenes.

## Environment Setup

This project uses `uv` for Python dependency management:

```bash
# Initialize project with Python 3.12+
uv init python=3.12

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

## Project Structure

- `main.py` - Entry point with basic hello world template
- `detect_image.py` - Core detection script that loads YOLOv8n model and runs inference on images
- `yolov8n.pt` - Pre-trained YOLOv8 nano model weights
- `pasto_vacas.png` - Sample input image (pasture with cows)
- `result_bouding_box.jpg` - Output image with detection bounding boxes

## Running Detection

To run object detection on an image:

```bash
python detect_image.py
```

This will:
1. Load the YOLOv8n model from `yolov8n.pt`
2. Process the image at `pasto_vacas.png`
3. Display results (if display available)
4. Save annotated image to `result_bouding_box.jpg`

## Architecture Notes

The project currently uses a simple direct inference approach:
- Model initialization: `YOLO("yolov8n.pt")` loads the pre-trained nano model
- Inference: `model(image_path)` returns detection results
- Results contain bounding boxes, class labels, and confidence scores
- The `results[0]` object provides `.show()` and `.save()` methods for visualization

The YOLOv8 nano model is optimized for speed and is the smallest variant in the YOLOv8 family, suitable for real-time applications on resource-constrained devices.
