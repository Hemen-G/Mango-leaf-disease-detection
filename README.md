# Mango Disease Detection

## Project Overview
This project implements a deep learning solution for detecting and classifying diseases in mango leaves using convolutional neural networks (CNNs). The system can identify 8 different conditions affecting mango plants, including both healthy leaves and various disease states.

## Features
- **Multi-class Classification**: Detects 8 different mango leaf conditions
- **High Accuracy**: Achieves 90% overall accuracy on validation data
- **Data Preprocessing**: Automated dataset splitting and image preprocessing
- **Model Visualization**: Includes training progress tracking and confusion matrix analysis
- **Export Capabilities**: Save trained models and training history for future use

## Supported Disease Classes
The model can identify the following conditions:
1. **Anthracnose** - Fungal disease causing dark, sunken lesions
2. **Bacterial Canker** - Bacterial infection leading to canker formation
3. **Cutting Weevil** - Pest damage from weevil infestations
4. **Die Back** - Progressive dying back of twigs and branches
5. **Gall Midge** - Pest-induced gall formations
6. **Healthy** - Disease-free mango leaves
7. **Powdery Mildew** - Fungal infection with white powdery coating
8. **Sooty Mould** - Fungal growth on honeydew secretions

## Model Architecture
The CNN model features:
- **Input Layer**: 256×256×3 RGB images
- **Convolutional Blocks**: 5 blocks with increasing filters (32→512)
- **Pooling Layers**: MaxPooling after each convolutional block
- **Dropout Layers**: For regularization (25% and 40% dropout rates)
- **Dense Layers**: 1024-unit hidden layer with 8-unit softmax output
- **Total Parameters**: ~23.6 million trainable parameters

## Performance Metrics
- **Overall Accuracy**: 90%
- **Precision**: 91% (macro average)
- **Recall**: 90% (macro average)
- **F1-Score**: 90% (macro average)

### Individual Class Performance:
- Cutting Weevil: 100% precision and recall
- Die Back: 98% precision and recall
- Healthy: 97% precision, 91% recall
- Anthracnose: 98% precision, 86% recall
- Bacterial Canker: 90% precision, 86% recall
- Gall Midge: 76% precision, 90% recall
- Powdery Mildew: 87% precision, 81% recall
- Sooty Mould: 81% precision, 90% recall

## Dataset
- **Total Images**: 3,998 mango leaf images
- **Training Set**: 2,798 images (70%)
- **Validation Set**: 800 images (20%)
- **Test Set**: 400 images (10%)
- **Image Size**: 256×256 pixels
- **Format**: RGB color images

## Installation & Requirements

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Google Colab (for original implementation)

### Required Libraries
```bash
pip install tensorflow matplotlib pillow scikit-learn seaborn pandas
```

## Usage

### 1. Data Preparation
```python
# Mount Google Drive and organize dataset
from google.colab import drive
drive.mount('/content/drive')

# Dataset will be automatically split into train/validation/test sets
```

### 2. Model Training
```python
# The model automatically trains with optimized parameters:
# - Learning rate: 0.0001
# - Batch size: 32
# - Epochs: 10
# - Optimizer: Adam
# - Loss: Categorical Crossentropy
```

### 3. Making Predictions
```python
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('/content/drive/MyDrive/mango_disease_model.keras')

# Prepare test images
test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/dataset/mango/test',
    image_size=(256, 256),
    batch_size=32
)

# Make predictions
predictions = model.predict(test_set)
```

## File Structure
```
Mango_Disease_Detection/
├── Mango_Disease_detection.ipynb     # Main notebook
├── mango_disease_model.keras         # Trained model
├── training_history.json            # Training metrics
├── dataset/
│   ├── train/                       # Training images
│   ├── validation/                  # Validation images
│   └── test/                        # Test images
└── README.md                        # This file
```

## Key Features Implemented

### Data Preprocessing
- Automatic dataset splitting (70/20/10)
- Image resizing to 256×256 pixels
- Data validation and integrity checks
- Class-balanced distribution

### Model Optimization
- Progressive convolutional architecture
- Dropout regularization to prevent overfitting
- Learning rate optimization
- Early stopping implementation

### Evaluation Metrics
- Comprehensive classification reports
- Confusion matrix visualization
- Precision, recall, and F1-score tracking
- Training/validation accuracy monitoring

## Results & Insights
- The model shows excellent performance on most disease classes
- Cutting Weevil detection achieves perfect accuracy
- Some confusion exists between similar-looking diseases
- The model generalizes well to unseen data

## Applications
- **Agricultural Monitoring**: Early detection of mango diseases
- **Farm Management**: Targeted treatment planning
- **Research**: Study of disease patterns and progression
- **Education**: Training tool for agricultural students

## Limitations & Future Work
- Limited to 8 specific disease classes
- Performance varies across different disease types
- Potential for improvement with more diverse datasets
- Could benefit from data augmentation techniques

## Contributors
This project demonstrates the application of deep learning in agricultural technology, specifically for plant disease detection and classification.

## License
This project is intended for educational and research purposes. Please ensure proper attribution when using or modifying the code.
