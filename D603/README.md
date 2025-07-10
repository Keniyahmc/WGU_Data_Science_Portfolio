# D603 – Machine Learning

## 📘 Course Overview
This course focused on applying **supervised**, **unsupervised**, and **time series modeling** techniques to real-world datasets to extract patterns and inform business decisions. Students learned to evaluate multiple model types (classification, clustering, forecasting), measure model performance, and justify model selection based on business needs and accuracy metrics.

---

## 🎯 Competencies Demonstrated
- Recommended supervised learning models based on comparative performance
- Recommended unsupervised clustering methods to uncover natural groupings
- Applied time series forecasting (ARIMA) to detect revenue trends
- Evaluated model performance using metrics such as AUC, silhouette score, and RMSE
- Justified modeling decisions based on business cases and technical evaluation

---

## 🛠 Tools & Technologies Used
- Python (pandas, NumPy, scikit-learn, statsmodels, matplotlib, seaborn)
- Jupyter Notebook
- GitLab (code version control and submission)

---

## 📂 Project Tasks

### Task 1: Supervised Learning – Random Forest Classification
- Predict patient readmission based on demographic and medical history
- Cleaned and encoded data, split into train/validate/test sets
- Tuned model using GridSearchCV and achieved ~98% accuracy
- Identified top readmission predictors (e.g., Age, Initial Admin, Diagnosis History)

📄 Files:
- `D603_Task1.2.pdf`
- `D603Task1.ipynb`

---

### Task 2: Unsupervised Learning – K-Means Clustering
- Clustered patients by demographics and medical history using K-means
- Used the elbow method and silhouette scores to determine **k=3**
- Scaled and encoded 50 features, then reduced dimensions using PCA for visualization
- Found meaningful groupings that could inform resource allocation and patient care

📄 Files:
- `D603_Task2.pdf`
- `D603Task2.ipynb`

---

### Task 3: Time Series Forecasting – Revenue Trends
- Forecasted hospital revenue using **ARIMA**
- Conducted stationarity tests (ADF), seasonal decomposition, and autocorrelation analysis
- Built and validated a time series model with low RMSE
- Provided actionable insight into future revenue patterns for strategic planning

📄 Files:
- `D603_Task3.pdf`
- `D603Task3.ipynb`

---

## ✅ Key Takeaways
- Applied and evaluated multiple machine learning algorithms across business use cases
- Developed a high-accuracy model to predict readmission, aiding preventative care planning
- Uncovered hidden groupings in medical data through clustering for tailored intervention
- Forecasted time-based revenue patterns for proactive business decisions
