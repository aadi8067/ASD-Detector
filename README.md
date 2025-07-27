# 🧠 Autism Spectrum Disorder (ASD) Detection using Machine Learning

A multi-algorithmic machine learning approach for accurately detecting Autism Spectrum Disorder (ASD) using behavioral and personal data.

---

## 📄 Research Paper Title

**Multi-Algorithmic Approach for Accurate Detection of Autism Spectrum Disorder: A Machine Learning Perspective**

---

## 👥 Authors

- Dr. A. K. Chaudhari  
- **Aditya A. Dhakane**
- Shreyas Kulkarni  
- Asawari Kshirsagar  
- Vaibhav Mhetre  
- Shreya Anjikhane  


---

## 📝 Abstract

This project utilizes a multi-algorithmic machine learning pipeline to detect Autism Spectrum Disorder using 704 patient records. Logistic Regression achieved 100% accuracy and outperformed other models like SVM, Random Forest, and KNN. The proposed system supports real-time web-based deployment and aims to assist clinicians in ASD screening.

---

## 📊 Dataset

- **Records:** 704 entries  
- **Features:** 21 (10 behavioral + 10 individual characteristics + target class)  
- **Target:** `ASD` (Yes/No)  
- 📥 [Dataset Source - Kaggle](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers)

---

## ⚙️ Models Implemented

| Algorithm                  | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression        | 100%     | 1.00      | 1.00   | 1.00     |
| Support Vector Machine     | 95.08%   | 0.94      | 0.94   | 0.94     |
| Random Forest              | 94.26%   | 0.93      | 0.94   | 0.94     |
| K-Nearest Neighbors (KNN)  | 93.44%   | 0.92      | 0.93   | 0.93     |

---

## 🧪 Evaluation Metrics

- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, F1-score  
- Feature Importance (from Random Forest)

---

## 🔍 Feature Engineering & Preprocessing

- Label Encoding & One-Hot Encoding  
- Missing value imputation  
- Min-Max Normalization & Z-score Standardization  
- SMOTE for class imbalance handling  

---

## 🧠 Feature Importance Highlights

Key behavioral features impacting diagnosis:
- `A9_Score`, `A6_Score`, `A5_Score`  
Less important features: Age and Ethnicity

---

## 📂 Project Structure

ASD-Detector/
│
├── models/
│ └── ASD_Model_Training.ipynb # Model training notebook (Colab)
├── app.py # Web app for ASD prediction
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── app.py ASD Detection Web App #A simple Flask-based web application (app.py) allows users to input symptoms and instantly receive ASD prediction results using the trained model.

---

## 🧪 Model Training (Colab)

Train and evaluate models using the provided notebook:

📁 `models/ASD_Model_Training.ipynb`  
🔗 [Open in Google Colab](https://colab.research.google.com/)

**Steps**:
1. Upload or load the dataset  
2. Run all cells to:
   - Preprocess data  
   - Train and evaluate 4 ML models  
   - Visualize metrics and feature importance  

---

## 💻 Run Locally using PyCharm or Any IDE

### 📌 Requirements:
- Python 3.7+
- Flask (if using app.py for deployment)

### 🔧 Setup Instructions:
1. Open the project folder in **PyCharm**
2. (Optional) Create a virtual environment
3. Install required packages:
    pip install -r requirements.txt

To run the application locally:
python app.py

Open browser and go to:
http://127.0.0.1:5000

---
---

### 🧠 Future Scope
Integrate Deep Learning (CNN, LSTM)

Eye-tracking / MRI-based detection

Federated Learning for privacy-preserving training

Mobile App deployment (TensorFlow Lite)

---
---

### 📄 License
This project is developed for academic and research purposes only under Vishwakarma Institute of Technology, Pune.

---
---
### 📌 Citation
Dhakane Aditya, et al. "Multi-Algorithmic Approach for Accurate Detection of Autism Spectrum Disorder: A Machine Learning Perspective." Vishwakarma Institute of Technology, Pune.

---
---
### 🙏 Thanks for exploring this project!
---