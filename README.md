# 🧠 Image Classification using CNN vs MobileNetV2

## 📌 Project Overview  
This project compares two deep learning approaches for image classification of products:
- A **Custom Convolutional Neural Network (CNN)** built from scratch  
- A **MobileNetV2-based Transfer Learning** model (🏆 Best Performing)

It aims to solve challenges related to **class imbalance**, **small dataset size**, and **overfitting**, using real-world product images from multiple categories.

---

## 📂 Project Resources  
🔹 **Dataset (Product Images)**: 1,909 images in 5 classes (Not uploaded due to size)  
🔹 **Project Code (.ipynb)**: [View Notebook](https://github.com/SunnyRao07/cnn-vs-mobilenet-image-classification/blob/main/image_classification_cnn_vs_mobilenetv2_code.ipynb)  
🔹 **Project Report (DOCX File)**: [Download Report](https://github.com/SunnyRao07/cnn-vs-mobilenet-image-classification/blob/main/cnn_vs_mobilenetv2_report.docx)

---

## 🧾 Dataset Overview  
- **Total Images**: 1,909  
- **Classes**: Product_1 to Product_5  
- **Images per class**:
  - Product_1: 510  
  - Product_2: 14 (⚠️ Highly Imbalanced)  
  - Product_3: 400  
  - Product_4: 385  
  - Product_5: 600  
- **Preprocessing**:
  - Resized to 224x224 pixels  
  - Normalized pixel values to [0,1]  
  - Real-time augmentation: rotation, shift, shear, zoom, flip  
  - Class weights for imbalance handling

---

## 🛠 Model Architectures

### 1️⃣ **Custom CNN**  
- Conv2D (32 → 64 → 128) + MaxPooling  
- Dense(128) + Dropout(0.5)  
- Softmax output layer  
- Optimizer: Adam (lr=1e-3)  
- EarlyStopping and ModelCheckpoint  
- Epochs: 25  

### 2️⃣ **MobileNetV2 Transfer Learning** (🏆 Best Model)  
- Pretrained on ImageNet (without top layers)  
- Classification head: GAP → Dense(128) → Dropout → Dense(5, softmax)  
- Training Strategy:
  - Phase 1: Freeze base, train head for 10 epochs  
  - Phase 2: Unfreeze last 20 layers, fine-tune for 5 epochs (lr=1e-5)  
- EarlyStopping enabled in both phases  

---

## 📈 Model Performance

| Model         | Accuracy | Test Loss | Remarks                          |
|---------------|----------|-----------|----------------------------------|
| Custom CNN    | 92.68%   | 0.25      | Overfitting observed             |
| **MobileNetV2** | **97.21%** | **0.07**   | Generalized well, faster training |

📌 **Note**: Both models failed to correctly classify `Product_2` due to very limited training data.

---

## 📊 Evaluation Metrics  
- **Accuracy**  
- **Loss**  
- **Precision, Recall, F1-Score (per class)**  
- **Confusion Matrix**  
- **Training & Validation Curves**

---

## 🧠 Key Observations  
- Transfer learning significantly improved accuracy and generalization  
- CNN showed signs of overfitting and required more memory  
- Severe class imbalance led to 0% F1 for Product_2 in both models  
- MobileNetV2 had ~165K trainable parameters vs CNN’s ~11M  

---

## 🚀 Future Scope  
🔹 Apply **advanced synthetic augmentation** (e.g., GANs) for minority classes  
🔹 Explore **EfficientNet** or **ResNet50** for better transfer performance  
🔹 Consider **active learning** for iterative data improvement  
🔹 Deploy the model via Flask or Streamlit for real-time classification

---

## 🏗 Project Highlights  
This project demonstrates a complete **deep learning pipeline** including:  
✔️ Data Preprocessing & Augmentation  
✔️ CNN & Transfer Learning Model Development  
✔️ Evaluation with Visualization  
✔️ Addressing Class Imbalance using Weights  
✔️ Academic Report & Documentation

---

## 👥 Authors  
- Sunny Rao Karegam  
- Sandeep Kumar Kandagatla  
- Srikanth Kannamoni  
- Alphin Stivi John  

🎓 MSc Data Analytics – Dublin Business School (2025)

---

## 📌 Note  
This is an academic project submitted as part of the **Machine Learning & Pattern Recognition** module.  
The dataset is not included in this repository due to space/privacy restrictions.  
Please contact the authors for access or use a similar open-source dataset for replication.

---
