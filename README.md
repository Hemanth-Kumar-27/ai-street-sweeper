# AI Street Sweeper

AI Street Sweeper is a prototype control system for an intelligent street sweeping application. It uses computer vision and machine learning to detect debris, segment the road area, and adapt brush and fan settings using fuzzy logic.

## Project Overview

The application is built with a Qt-based GUI using `PySide6` and processes a video stream to:
- display the original video feed
- compute a road region of interest (ROI)
- perform semantic road segmentation with SegFormer
- classify debris levels using a MobileNetV2 debris classifier
- compute adaptive brush and fan RPM settings with a fuzzy logic controller
- estimate power usage and energy savings

## Repository Structure

- `app.py` - main GUI application and video processing loop
- `config.py` - application configuration, paths, model settings, and constants
- `requirements.txt` - Python dependencies
- `utils/`
  - `video.py` - video capture, ROI extraction, and CNN ROI preprocessing
  - `predictor.py` - MobileNetV2 debris prediction
  - `segmentation.py` - SegFormer semantic segmentation and road ROI extraction
  - `controller.py` - fuzzy logic controller for brush/fan settings
  - `power.py` - adaptive power and energy saving calculations
- `model/`
  - `mobilenetv2_debris_classifier.keras` - trained debris classification model
  - `class_names.json` - class labels for the debris classifier
  - `segformer_b0/` - pretrained SegFormer model files for segmentation
- `videos/` - video inputs for the demo application
- `Test.py` - sample script demonstrating SegFormer road segmentation

## Dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- `tensorflow`
- `opencv-python`
- `numpy`
- `Pillow`
- `PySide6`
- `transformers`
- `torch`
- `torchvision`
- `scikit-fuzzy`
- `matplotlib`
- `tqdm`

## Configuration

Update `config.py` before running the application if necessary:
- `VIDEO_PATH` - path to the input video file
- `MODEL_PATH` - path to the trained MobileNetV2 model
- `CLASS_NAMES_PATH` - path to the class label file
- `SEGFORMER_MODEL` - path to the pretrained SegFormer model directory
- `FRAME_WIDTH`, `FRAME_HEIGHT`, `FPS` - video processing settings
- `MAX_BRUSH_RPM`, `MAX_FAN_RPM`, `MAX_POWER` - machine control limits

## Running the Application

From the project root, run:

```bash
python app.py
```

This launches the AI Street Sweeper control system UI and begins processing frames from the configured video.

## Running the Segmentation Test

`Test.py` is a simple demo script that:
- loads an image from `cnn_test.jpg`
- runs SegFormer road segmentation
- prints coverage statistics
- displays the original image, road mask, and extracted road ROI

Run it with:

```bash
python Test.py
```

## Notes

- The current `config.py` includes an absolute path for `VIDEO_PATH`. Update this path if your repository is moved or if you want to test a different video.
- The GUI updates predictions and segmentation results periodically (every 15 frames).
- The fuzzy logic controller maps debris severity to brush and fan RPM settings.

## Future Improvements

Potential enhancements include:
- support for live camera input
- dynamic ROI tuning
- more advanced debris classification and segmentation models
- using the `get_cnn_roi` method from `utils/video.py` for direct CNN input preprocessing
- adding logging and error handling for missing model/data files
