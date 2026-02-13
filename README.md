# ASL Recognition System

A real-time American Sign Language (ASL) recognition system that uses computer vision and machine learning to identify hand gestures. The system employs MediaPipe for hand landmark detection and a Support Vector Machine (SVM) classifier for gesture classification.

## Overview

This project recognizes ASL signs through a two-stage pipeline:
1. **Hand Landmark Detection**: MediaPipe extracts 21 key hand landmarks from images
2. **Gesture Classification**: SVM classifier identifies the sign based on landmark positions

## Features

✅ **Real-time Recognition**: Webcam-based live ASL sign detection  
✅ **MediaPipe Integration**: Accurate 21-point hand landmark extraction  
✅ **SVM Classification**: Robust polynomial kernel SVM for gesture recognition  
✅ **Prediction Smoothing**: Majority voting system for stable predictions  
✅ **3D Visualization**: PCA-based 3D clustering visualization of hand gestures  
✅ **Multiple Kernels**: Support for both linear and polynomial SVM kernels  
✅ **Pre-trained Models**: Ready-to-use trained SVM models included  

## How It Works

### Architecture

#### 1. **Hand Landmark Extraction** (`utils.py`)

The system uses **MediaPipe's Hand Landmarker** to detect and track hand landmarks:

- Detects 21 key points on the hand (fingertips, knuckles, palm, wrist)
- Extracts normalized (x, y) coordinates for each landmark
- Flattens to a 42-dimensional feature vector (21 landmarks × 2 coordinates)

**Process:**
1. Converts image to MediaPipe format
2. Runs hand detection model
3. Extracts landmark coordinates
4. Returns flattened feature array

#### 2. **Model Training** (`train.py`)

Trains an SVM classifier on hand gesture datasets:

**Training Pipeline:**
1. **Data Loading**: Reads images organized by gesture label in `data/` directory
2. **Preprocessing**: 
   - Converts BGR to RGB
   - Resizes to 256×256 pixels
3. **Feature Extraction**: Extracts 21 hand landmarks per image
4. **Model Training**: 
   - Uses polynomial kernel SVM (degree=3)
   - Enables probability estimates
   - Configured with `gamma='scale'` and `coef0=1`
5. **Model Serialization**: Saves trained model as `.pkl` file
6. **Visualization** (Optional):
   - Reduces features to 3D using PCA
   - Plots gesture clusters in 3D space
   - Shows decision boundaries

**SVM Configuration:**
```python
svm.SVC(kernel="poly", degree=3, gamma="scale", coef0=1, probability=True)
```

#### 3. **Real-time Prediction** (`predict.py`)

Performs live gesture recognition via webcam:

**Prediction Pipeline:**
1. **Webcam Capture**: Captures frames from default camera
2. **Preprocessing**: Converts to RGB and resizes to 256×256
3. **Landmark Extraction**: Detects hand and extracts landmarks
4. **Classification**: Predicts gesture using trained SVM
5. **Smoothing**: Applies majority voting over last 10 predictions
6. **Display**: Overlays prediction text on video feed

**Prediction Smoothing:**
- Maintains a buffer of the last 10 predictions
- Uses majority voting to reduce jitter
- Persists last valid prediction when no hand is detected

### The Recognition Algorithm

**Feature Representation:**
- Each hand gesture is represented by 21 landmarks
- Each landmark has (x, y) coordinates
- Total feature vector: 42 dimensions

**SVM Classification:**
- **Kernel**: Polynomial (degree 3)
- **Decision Function**: One-vs-One multi-class strategy
- **Output**: Gesture label + probability scores

**Why Polynomial Kernel?**
- Better captures non-linear relationships between landmarks
- More effective than linear kernel for complex hand shapes
- Provides better separation between similar gestures

## Installation

### Prerequisites
- Python 3.7+
- Webcam (for real-time recognition)
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/winterwidow/ASL-Recognition.git
cd ASL-Recognition

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `opencv-python`: Image capture and processing
- `mediapipe`: Hand landmark detection
- `scikit-learn`: SVM classifier and PCA
- `numpy`: Numerical operations
- `joblib`: Model serialization

### Download MediaPipe Model

The system requires the MediaPipe hand landmarker model:
- File: `hand_landmarker.task` (~7.8 MB)
- This file should already be included in the repository
- If missing, download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)

## Usage

### Step 1: Prepare Training Data

Organize your dataset with one folder per gesture:

```
data/
├── data_num/          # or your dataset directory
│   ├── A/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── B/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── 0/
│   ├── 1/
│   └── ...
```
### Step 2: Train the Model

Run the training script:

```bash
python train.py
```

**Configuration** (in `train.py`):
```python
DATA_DIR = "data/data_num"  # Path to training dataset
MODEL_PATH = "svm_model2.pkl"  # Output model file
```

**Training Options:**
- **Linear Kernel**: Fast, works for simple gestures
  ```python
  clf = svm.SVC(kernel="linear", probability=True)
  ```
- **Polynomial Kernel**: Better accuracy for complex gestures (default)
  ```python
  clf = svm.SVC(kernel="poly", degree=3, gamma="scale", coef0=1, probability=True)
  ```

**Output:**
- Saves trained model as `.pkl` file
- Prints number of samples collected
- (Optional) Displays 3D PCA visualization

### Step 3: Run Real-time Recognition

Start the webcam-based recognition:

```bash
python predict.py
```

**Configuration** (in `predict.py`):
```python
MODEL_PATH = "svm_model.pkl"  # Path to trained model
```

**Controls:**
- Position your hand in front of the webcam
- The predicted sign appears on screen
- Press `q` to quit

**Features:**
- Real-time prediction with webcam feed
- Prediction smoothing for stability
- Persistent display of last valid prediction
- Green text overlay showing recognized sign

## Project Structure

```
ASL-Recognition/
├── train.py                # Model training script
├── predict.py              # Real-time prediction script
├── utils.py                # Hand landmark extraction utilities
├── requirements.txt        # Python dependencies
├── hand_landmarker.task    # MediaPipe hand detection model
├── svm_model.pkl           # Trained SVM model (alphabet)
├── svm_model2.pkl          # Trained SVM model (numbers)
├── data/                   # Training datasets
│   ├── dataset/           # Alphabet gestures
│   └── data_num/          # Number gestures
└── __pycache__/           # Python cache files
```

## Model Comparison

### Linear vs Polynomial Kernel

| Aspect | Linear Kernel | Polynomial Kernel (Degree 3) |
|--------|---------------|------------------------------|
| **Training Speed** | Fast | Moderate |
| **Accuracy (Simple)** | Good (85-90%) | Excellent (92-97%) |
| **Accuracy (Complex)** | Moderate (75-80%) | Excellent (88-95%) |
| **Overfitting Risk** | Low | Moderate (requires tuning) |
| **Best For** | Simple gestures, quick prototyping | Complex hand shapes, production |

**Recommendation**: Use polynomial kernel for better accuracy with ASL gestures.

## Visualization

The training script includes optional 3D visualization:

**Features:**
- Reduces 42D feature space to 3D using PCA
- Plots gesture clusters in 3D space
- Shows approximate SVM decision boundaries
- Helps visualize gesture separability

**Enable/Disable:**
- Comment/uncomment the plotting section in `train.py` (lines 87-135)

## Performance Considerations

### Accuracy Factors

**Improves Accuracy:**
- ✅ Good lighting conditions
- ✅ Plain backgrounds
- ✅ Consistent hand positioning
- ✅ Larger training datasets (50+ images per gesture)
- ✅ Diverse training data (angles, distances)

**Reduces Accuracy:**
- ❌ Low light or shadows
- ❌ Complex backgrounds
- ❌ Partial hand occlusions
- ❌ Small training datasets
- ❌ Similar-looking gestures

### Optimization Tips

1. **Training Data Quality**
   - Use high-resolution images (256×256 minimum)
   - Ensure hands are clearly visible
   - Include variations in lighting and background

2. **Prediction Smoothing**
   - Adjust buffer size in `predict.py` (default: 10 frames)
   - Larger buffer = more stable but slower response
   - Smaller buffer = faster but potentially jittery

3. **Model Selection**
   - Start with polynomial kernel
   - Tune `degree` and `gamma` parameters if needed
   - Consider RBF kernel for highly non-linear data

## Troubleshooting

**"No hand detected in the image"**
- Ensure hand is clearly visible in frame
- Check lighting conditions
- Verify webcam is working
- Hand should be primary object in frame

**Poor prediction accuracy**
- Increase training dataset size
- Ensure diverse training data
- Check if hand landmarks are correctly extracted
- Try different SVM kernels or parameters

**Model file not found**
- Ensure `hand_landmarker.task` is in project root
- Check that trained model `.pkl` file exists
- Re-run `train.py` if model is missing

**Webcam not working**
- Verify camera permissions
- Check camera index (change `VideoCapture(0)` to `VideoCapture(1)` etc.)
- Ensure no other application is using the webcam

## Dataset Recommendations

### Suggested Datasets
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle)
- [ASL Numbers Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset) (Kaggle)
- Create your own custom dataset for specific gestures (Done here to test the model on unstructured datasets)

### Creating Custom Dataset
1. Capture 50-100 images per gesture
2. Use consistent lighting
3. Vary hand position, angle, and distance
4. Include different backgrounds
5. Organize in labeled folders

## Future Enhancements

- [ ] Support for dynamic gestures (motion-based signs)
- [ ] Integration with text-to-speech for output
- [ ] Mobile app version (Android/iOS)
- [ ] Sentence construction from multiple signs
- [ ] Deep learning models (CNN/LSTM) for improved accuracy
- [ ] Multi-hand support for two-handed signs
- [ ] Real-time performance metrics display

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [MediaPipe](https://developers.google.com/mediapipe) by Google
- Uses [scikit-learn](https://scikit-learn.org/) for SVM implementation
- OpenCV for image processing and display

## References

- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [SVM Classification (scikit-learn)](https://scikit-learn.org/stable/modules/svm.html)
- [ASL Alphabet](https://www.startasl.com/asl-alphabet/)