# Mango Disease Detection

## Project Overview
A comprehensive deep learning solution for detecting and classifying diseases in mango leaves. This project includes both model training and a ready-to-use demo for real-time disease prediction.

## 📁 Project Structure
```
Mango_Disease_Detection/
├── Mango_Disease_detection.ipynb          # Main training notebook
├── demo_for_mango_disease_detection_.ipynb # Demo & inference notebook
├── mango_disease_model.keras              # Pre-trained model (90MB)
├── training_history.json                  # Training metrics & history
├── dataset/
│   ├── train/                            # 2,798 training images (70%)
│   ├── validation/                       # 800 validation images (20%)
│   └── test/                             # 400 test images (10%)
└── README.md                             # Project documentation
```

## 🚀 Quick Start - Demo

### Run the Demo Notebook:
The `demo_for_mango_disease_detection_.ipynb` provides a complete inference pipeline:

```python
# Load pre-trained model
from tensorflow import keras
model_path = '/content/drive/MyDrive/mango_disease_model.keras'
model = keras.models.load_model(model_path)

# Test the model
model.summary()
```

### Make Predictions:
```python
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load your mango leaf image
image_path = "/path/to/your/mango_leaf.jpg"

# The demo will automatically:
# 1. Load and display the image
# 2. Preprocess it for the model
# 3. Make a prediction
# 4. Show the disease name and confidence
```

## 🎯 Supported Disease Classes
The model detects 8 different conditions:

| Class Index | Disease Name | Description |
|-------------|--------------|-------------|
| 0 | **Anthracnose** | Fungal disease causing dark lesions |
| 1 | **Bacterial Canker** | Bacterial infection with canker formation |
| 2 | **Cutting Weevil** | Pest damage from weevil infestations |
| 3 | **Die Back** | Progressive dying of twigs and branches |
| 4 | **Gall Midge** | Pest-induced gall formations |
| 5 | **Healthy** | Disease-free mango leaves |
| 6 | **Powdery Mildew** | Fungal infection with white coating |
| 7 | **Sooty Mould** | Fungal growth on honeydew |

## 📊 Model Performance

### Overall Metrics:
- **Accuracy**: 90%
- **Precision**: 91% (macro average)
- **Recall**: 90% (macro average)
- **F1-Score**: 90% (macro average)

### Per-Class Performance:
| Disease | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Anthracnose | 98% | 86% | 91% |
| Bacterial Canker | 90% | 86% | 88% |
| Cutting Weevil | 100% | 100% | 100% |
| Die Back | 98% | 98% | 98% |
| Gall Midge | 76% | 90% | 83% |
| Healthy | 97% | 91% | 94% |
| Powdery Mildew | 87% | 81% | 84% |
| Sooty Mould | 81% | 90% | 85% |

## 🏗️ Model Architecture

### Technical Specifications:
- **Input Size**: 256 × 256 × 3 (RGB images)
- **Architecture**: Sequential CNN with 5 convolutional blocks
- **Filters**: 32 → 64 → 128 → 256 → 512 (progressive increase)
- **Dropout**: 25% (convolutional), 40% (dense)
- **Output**: 8-class softmax classification

### Model Summary:
```
Total params: 70,787,450 (270.03 MB)
Trainable params: 23,595,816 (90.01 MB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 47,191,634 (180.02 MB)
```

## 📋 Dataset Information

### Distribution:
- **Total Images**: 3,998
- **Training Set**: 2,798 images (70%)
- **Validation Set**: 800 images (20%)
- **Test Set**: 400 images (10%)

### Class Balance:
- Each class contains approximately 500 images
- Balanced distribution across all 8 categories
- High-quality labeled images

## ⚙️ Installation & Setup

### Prerequisites:
- Python 3.7+
- TensorFlow 2.x
- Google Colab (recommended) or local environment

### Required Libraries:
```bash
pip install tensorflow matplotlib pillow scikit-learn seaborn pandas opencv-python numpy
```

## 🛠️ Usage Instructions

### Option 1: Demo & Predictions (Recommended)
1. Open `demo_for_mango_disease_detection_.ipynb`
2. Ensure `mango_disease_model.keras` is available
3. Modify `image_path` to point to your mango leaf image
4. Run all cells for instant predictions

### Option 2: Full Training
1. Open `Mango_Disease_detection.ipynb`
2. Mount your Google Drive with the dataset
3. Run all cells to train from scratch
4. Model will be saved automatically

### Option 3: Custom Integration
```python
# Integrate into your own code
from tensorflow import keras
import cv2
import numpy as np

class MangoDiseaseDetector:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.class_names = [
            'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 
            'Die Back', 'Gall Midge', 'Healthy', 
            'Powdery Mildew', 'Sooty Mould'
        ]
    
    def predict(self, image_path):
        # Preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(256, 256)
        )
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        
        # Make prediction
        prediction = self.model.predict(input_arr)
        result_index = np.argmax(prediction)
        
        return {
            'disease': self.class_names[result_index],
            'confidence': float(np.max(prediction)),
            'all_predictions': prediction[0].tolist()
        }
```

## 📈 Training Details

### Optimization:
- **Optimizer**: Adam (learning_rate=0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 10
- **Metrics**: Accuracy

### Training Progress:
- Epoch 1-3: Rapid accuracy improvement (25% → 82%)
- Epoch 4-7: Refinement and stabilization (82% → 94%)
- Epoch 8-10: Fine-tuning and overfitting prevention

## 🎮 Demo Features

### What the Demo Provides:
- ✅ Complete inference pipeline
- ✅ Real-time image preprocessing
- ✅ Visual results with disease name
- ✅ Confidence scores
- ✅ Easy image path configuration
- ✅ Model architecture inspection

### Example Output:
```
Predicted Disease: Anthracnose
Confidence: 95.23%
```

## 🌟 Key Features

### For End Users:
- **User-Friendly**: Simple demo notebook for instant predictions
- **High Accuracy**: 90% reliable disease detection
- **Fast Inference**: Quick predictions on new images
- **Visual Results**: Clear display of input and output

### For Developers:
- **Modular Code**: Easy to integrate into applications
- **Well-Documented**: Clear code structure and comments
- **Extensible**: Easy to add new disease classes
- **Reproducible**: Complete training pipeline provided

## 🚨 Limitations & Considerations

### Current Limitations:
- Limited to 8 specific disease classes
- Requires clear, well-lit leaf images
- Performance varies with image quality
- Model trained on specific mango varieties

### Image Requirements:
- **Format**: JPG, JPEG, or PNG
- **Size**: Minimum 256×256 pixels (auto-resized)
- **Quality**: Clear, focused images of individual leaves
- **Background**: Plain backgrounds work best

## 🔮 Future Enhancements

### Planned Improvements:
- Expand to more disease classes
- Mobile app development
- Real-time camera integration
- Multi-plant species support


### Research Directions:
- Transfer learning with newer architectures
- Data augmentation techniques
- Ensemble methods for improved accuracy
- Explainable AI for prediction interpretability

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional disease classes
- Performance optimization
- Documentation improvements
- Translation to other languages

## 📝 Citation

If you use this project in your research, please cite:
```
Mango Disease Detection using Deep Learning (2024)
CNN-based classification of 8 mango leaf diseases
Accuracy: 90%, Dataset: 3,998 images
```

## 📞 Support

For questions or issues:
1. Check the demo notebook for usage examples
2. Review the training notebook for technical details
3. Ensure all dependencies are properly installed

## 📄 License

This project is intended for educational and research purposes. Please ensure proper attribution when using or modifying the code.

---

**Ready to detect mango diseases?** Run `demo_for_mango_disease_detection_.ipynb` to get started immediately!
