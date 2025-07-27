🧠 Autism Spectrum Disorder Detection using Machine Learning
A multi-algorithmic machine learning system to detect Autism Spectrum Disorder (ASD) based on behavioral and personal traits.

📄 Paper Title
Multi-Algorithmic Approach for Accurate Detection of Autism Spectrum Disorder: A Machine Learning Perspective

👨‍🔬 Developed By-
    Aditya Dhakane

📚 Abstract
This project uses machine learning models — Logistic Regression, SVM, Random Forest, and KNN — to detect ASD. Logistic Regression achieved the best performance with 100% accuracy. The system supports real-time screening through a web-based interface.

📊 Dataset
Total Records: 704
Features: 21 (10 behavioral + 10 individual traits)
Target: ASD Diagnosed (Yes/No)

⚙️ Machine Learning Models Used
        Model	    Accuracy	Precision	Recall	F1-Score
Logistic Regression	100%	    1.00	    0.00    1.00
SVM	                95.08%	    0.94	    0.94	0.94
Random Forest	    94.26%	    0.93	    0.94	0.94
KNN	                93.44%	    0.92	    0.93    0.93

🧪 Evaluation Metrics
Confusion Matrix

Accuracy

Precision, Recall, F1-Score

Feature Importance (from Random Forest)

📈 Feature Importance Highlights
Most important features for ASD detection:

A9_Score, A6_Score, A5_Score (behavioral traits)

Age and Ethnicity were less important

🧠 Preprocessing Techniques
Imputation for missing values

One-hot encoding & label encoding

Normalization (Min-Max) and Standardization (Z-score)

SMOTE for handling class imbalance

🖥 Deployment
Colab notebooks and backend logic are designed for real-time web deployment using Flask or Streamlit.

🚀 How to Run 
If you have a .ipynb:
Open in Google Colab
Upload the dataset if required
Run each cell sequentially

📎 Future Enhancements
Integrate Deep Learning (CNN, LSTM)
Add MRI/Eye-tracking data
Mobile/Cloud deployment with federated learning

📄 License
This project is for academic and research use only.

🧠 Citation
Dhakane Aditya, "Multi-Algorithmic Approach for Accurate Detection of Autism Spectrum Disorder: A Machine Learning Perspective." Vishwakarma Institute of Technology, Pune.

📥 Download Dataset & Files
    https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers