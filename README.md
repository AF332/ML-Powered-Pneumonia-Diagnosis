# ML-powered Pneumonia Diagnosis 🩺📊

## Overview
This project leverages machine learning to classify chest X-ray images into three categories: **Normal, Viral Pneumonia, and Bacterial Pneumonia**. The objective is to develop an interpretable deep learning model that can assist radiologists in diagnosing pneumonia more efficiently. The project employs **Convolutional Neural Networks (CNNs)** along with **Grad-CAM** visualization techniques to explain model decisions.

## Table of Contents
- Dataset
- Data Preprocessing
- Model Training & Performance
- Model Explainability
- Comparison with Expert Radiologists
- Future Improvements
- Installation & Usage
- Results
- Key Takeaways

## Dataset 📂
The dataset used for this project consists of **5,856** chest X-ray images divided into three classes:
- **Normal:** 1,266 images
- **Viral Pneumonia:** 1,194 images
- **Bacterial Pneumonia:** 2,224 images

### Challenges:
- **Class Imbalance:** The dataset has a higher number of bacterial pneumonia cases, which may bias the model.
- **Medical Image Complexity:** X-ray images require advanced feature extraction techniques to differentiate subtle patterns.

## Data Preprocessing ⚙️
1. **Image Resizing** - Standardized input size for CNN models.
2. **Normalization** - Pixel values scaled to the range [0,1].
3. **Data Augmentation** - Applied rotation, flipping, and contrast adjustments to balance class distribution.
4. **Expanded Dimensions** - Adapted images for compatibility with the chosen CNN architecture.

## Model Training & Performance 📊
### Training Strategy
- **CNN Architecture**: Combination of **VGG16 feature extraction** (first two layers) and a custom CNN model.
- **Batch Size**: 16 (for better generalization)
- **Epochs**: 90 (for stable learning)
- **Loss Function**: Categorical Cross-Entropy (since it's a multi-class classification task)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and AUC (Area Under Curve)

### Performance Metrics
| Metric | Training | Testing |
|--------|---------|---------|
| Accuracy | 82% | 80% |
| Precision | 80% | 79% |
| Recall | 85% | 80% |
| F1-score | 81% | 79% |
| AUC | 93% | 91% |

- The model **performs well on normal cases** but struggles with differentiating bacterial vs. viral pneumonia.
- **Overfitting Risk:** Minor overfitting observed, requiring additional regularization techniques.

## Model Explainability 🧐
To enhance transparency, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was used to visualize areas of interest in X-ray images. This technique highlights which regions influenced the model's decision, making it more interpretable for medical professionals.

## Comparison with Expert Radiologists 🏥
- The model’s outputs were **validated against radiologist annotations**.
- **Findings**: While the model performed well overall, it struggled with **viral pneumonia**, often confusing it with bacterial pneumonia.
- Expert feedback suggests **further dataset expansion** and **hybrid AI-radiologist workflows** to improve real-world applicability.

## Future Improvements 🚀
- **Increase dataset size** by incorporating more diverse chest X-ray images.
- **Train for more epochs** with fine-tuned hyperparameters.
- **Integrate additional feature extraction methods** to enhance classification performance.
- **Deploy a real-time demo application** showcasing model predictions.

## Installation & Usage ⚡
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python
```

### Running the Model
```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('pneumonia_model.h5')

# Preprocess input image
img = cv2.imread('sample_xray.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224)) / 255.0
img = np.expand_dims(img, axis=[0, -1])

# Make prediction
prediction = model.predict(img)
print("Predicted Class:", np.argmax(prediction))
```

## Results 📌
- The **CNN model achieved an 80% test accuracy**, with **Grad-CAM providing interpretability**.
- **Challenges:** The model struggles with bacterial vs viral pneumonia, highlighting the need for more balanced data.
- **Expert radiologists' insights** were incorporated to improve model reliability.

## Key Takeaways 🎯
- **AI can assist radiologists** but should be used alongside human expertise.
- **Model explainability (Grad-CAM) is crucial** for adoption in healthcare.
- **A larger, diverse dataset** can improve model generalization and reliability.

---
🔬 *This project demonstrates the potential of machine learning in medical imaging, showcasing both strengths and areas for future improvement.* 🚀
