
# Face Age and Gender Recognition Using OpenCV

This project demonstrates real-time face detection along with age and gender prediction using OpenCV and pre-trained deep learning models. The webcam feed detects faces, draws bounding boxes around them, and annotates each face with the predicted age group and gender.

## Features
- Real-time face detection using Haar Cascade Classifier.
- Age prediction using a deep learning model.
- Gender prediction using a deep learning model.
- Simple to set up and run on macOS with AVFoundation backend for the webcam.

## Prerequisites

Before running the project, ensure you have the following dependencies installed:

1. **Python 3.x**
2. **OpenCV** (with DNN module for deep learning support)

You can install OpenCV by running:
```bash
pip install opencv-python
```

## Pre-trained Models

You will need to download the following pre-trained models for age and gender prediction.

1. **Age Model:**
   - [deploy_age.prototxt](https://github.com/spmallick/learnopencv/tree/master/AgeGender)
   - [age_net.caffemodel](https://github.com/spmallick/learnopencv/tree/master/AgeGender)

2. **Gender Model:**
   - [deploy_gender.prototxt](https://github.com/spmallick/learnopencv/tree/master/AgeGender)
   - [gender_net.caffemodel](https://github.com/spmallick/learnopencv/tree/master/AgeGender)

Download these files and place them in the same directory as your Python script.

## Project Structure

```
/path/to/project
│
├── livecam.py                   # Main Python script
├── deploy_age.prototxt           # Age model architecture
├── age_net.caffemodel            # Age model weights
├── deploy_gender.prototxt        # Gender model architecture
├── gender_net.caffemodel         # Gender model weights
└── README.md                     # Project README
```

## Usage

1. **Clone the repository** or download the Python script (`livecam.py`).

2. **Download the model files** (as described above) and place them in the same directory as `livecam.py`.

3. **Run the script**:

```bash
python livecam.py
```

4. **Webcam Feed**: 
   - The webcam feed will open, and for each detected face, a bounding box will be drawn with age and gender predictions displayed at the top of the box.
   - Press **'q'** to close the webcam feed.

## Requirements

- OpenCV (Install via pip: `pip install opencv-python`)
- Python 3.x

## Troubleshooting

### "Could not open webcam" Error
- Ensure that your webcam has permission to be accessed by the terminal or IDE.
- Go to **System Preferences > Security & Privacy > Privacy > Camera**, and make sure your terminal or IDE is checked.

### "Failed to capture video frame" Error
- Make sure the webcam is not in use by another application.
- Ensure that you're using the correct OpenCV backend for macOS (`cv2.CAP_AVFOUNDATION`).

### File Not Found Errors
- Ensure that you have downloaded the `prototxt` and `caffemodel` files and placed them in the same directory as your script.
- Double-check the file paths in the Python script if they are in a different directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Pre-trained age and gender models from the [OpenCV DNN module](https://github.com/spmallick/learnopencv).
- OpenCV's Haar Cascade for face detection.

---

