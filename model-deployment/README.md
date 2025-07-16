# D602 – Deployment

## 📘 Course Overview
This course focused on **operationalizing machine learning models** within a business context. Key topics included analyzing business requirements, building scalable and secure pipelines, deploying predictive models, and implementing APIs to integrate insights into broader systems. The course emphasized real-world implementation strategies using MLOps principles.

---

## 🎯 Competencies Demonstrated
- Analyzed organizational constraints and requirements for model deployment
- Designed and implemented a pipeline for a machine learning model using MLflow
- Developed a FastAPI-based interface to expose the trained model
- Created automated tests to validate the API’s performance
- Deployed a data product for real-time predictions

---

## 🛠 Tools & Technologies Used
- Python (pandas, scikit-learn, matplotlib)
- MLflow for model tracking and experiment logging
- FastAPI for API development
- Pytest for test automation
- Jupyter Notebook & CSV data
- Pickle, JSON, matplotlib

---

## 📂 Project Tasks

### Task 1: Business Case Analysis
- Evaluated **Kronkers’ need for MLOps** to consolidate efforts across departments.
- Outlined **technical and organizational challenges** such as tool compatibility, team skill gaps, and limited budget.
- Recommended a centralized, version-controlled, secure, and scalable MLOps framework.

📄 File: `D602_Task_1.pdf`

---

### Task 2: ML Pipeline & Experiment Tracking
- Preprocessed flight delay data and applied feature engineering.
- Trained a **Polynomial Ridge Regression model** with different alpha values.
- Tracked experiments and metrics using **MLflow**.
- Saved the final model and **airport encodings** for deployment.

📄 File: `D602_Task_2.pdf`  
📄 Script: `Task_2_MLflow.py`

---

### Task 3: API Deployment & Docker Containerization
- Built a **FastAPI** app (`api.py`) to expose a delay prediction model via RESTful endpoints.
- Developed a `/predict/delays` route to accept query parameters: `departure_airport`, `arrival_airport`, `departure_time`, and `arrival_time`.
- Loaded and used a Ridge Regression model and one-hot encoded airport vectors for prediction.
- Included error handling for invalid inputs and formatted prediction output.
- Packaged and published the model and API using **Docker**, deploying the container to the WGU GitLab registry.

📄 Files:
- `D602_Task_3.pdf`
- `api.py`
- `test_api.py`
- `D602Task2C.ipynb` (supporting pipeline)
🧱 ![Container Image](container%20image%20(3).png)

---

## ✅ Key Takeaways
- Developed a robust machine learning pipeline using production-level tools
- Implemented **version control, artifact logging, and parameter tracking** with MLflow
- Created a functioning REST API for real-time model interaction
- Practiced deployment strategies aligned with enterprise-scale MLOps workflows
