#!/usr/bin/env python
# coding: utf-8

# In[38]:


#D1:Import Libraries and Set Parameters

import pandas as pd
import numpy as np
import datetime
import logging
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# In[40]:


# ✅ Manually set parameters
num_alphas = 20  # Number of Ridge alpha values to test
order = 1        # Degree of the polynomial features


# In[42]:


# ---- Logging Setup ----
logname = "polynomial_regression.txt"
logging.basicConfig(filename=logname, filemode='w',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)


# In[44]:


#D.2: Load Cleaned Data and Format Time
df = pd.read_csv("cleaned_data.csv")
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
month = df['MONTH'].iloc[0]
year = df['YEAR'].iloc[0]


# In[46]:


# ---- Format Time Columns ----
def format_hour(val):
    if pd.isnull(val): return np.nan
    val = int(val)
    if val == 2400: val = 0
    h = int(str(val).zfill(4)[:2])
    m = int(str(val).zfill(4)[2:])
    return datetime.time(h, m)

df['SCHEDULED_DEPARTURE'] = df['SCHEDULED_DEPARTURE'].apply(format_hour)
df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_ARRIVAL'].apply(format_hour)

def combine_datetime(row, col):
    if pd.isnull(row['DATE']) or pd.isnull(row[col]): return np.nan
    return datetime.datetime.combine(row['DATE'], row[col])

df['SCHEDULED_DEPARTURE'] = df.apply(lambda x: combine_datetime(x, 'SCHEDULED_DEPARTURE'), axis=1)
df['SCHEDULED_ARRIVAL'] = df.apply(lambda x: combine_datetime(x, 'SCHEDULED_ARRIVAL'), axis=1)


# In[48]:


#D.3: Split Data and Prepare Features ---- Split Data ----
df_train = df[df['SCHEDULED_DEPARTURE'].dt.day <= 21]
df_test = df[df['SCHEDULED_DEPARTURE'].dt.day > 21]


# In[50]:


# ---- Feature Engineering ----
def prepare_data(data):
    data = data.dropna(subset=['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_DELAY'])
    data = data[data['DEPARTURE_DELAY'] < 60]  # Remove outliers
    data['hour_depart'] = data['SCHEDULED_DEPARTURE'].dt.hour * 60 + data['SCHEDULED_DEPARTURE'].dt.minute
    data['hour_arrive'] = data['SCHEDULED_ARRIVAL'].dt.hour * 60 + data['SCHEDULED_ARRIVAL'].dt.minute
    return data[['hour_depart', 'hour_arrive', 'DEST_AIRPORT', 'DEPARTURE_DELAY']]

df_train_feat = prepare_data(df_train)
df_test_feat = prepare_data(df_test)


# In[52]:


# ---- One-Hot Encoding ----
le = LabelEncoder()
df_train_feat['DEST_AIRPORT_ENC'] = le.fit_transform(df_train_feat['DEST_AIRPORT'])
onehot = OneHotEncoder(sparse_output=False)
dest_encoded = onehot.fit_transform(df_train_feat[['DEST_AIRPORT_ENC']])

X_train = np.hstack((dest_encoded, df_train_feat[['hour_depart', 'hour_arrive']].values))
y_train = df_train_feat['DEPARTURE_DELAY'].values.reshape(-1, 1)


# In[54]:


# ---- Train/Validate Split ----
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)


# In[56]:


#D.4: Train Model & Track with MLflow ---- MLFlow Experiment ----
best_score = float("inf")
best_alpha = None
best_model = None
poly = PolynomialFeatures(degree=order)

mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    for i in range(num_alphas):
        alpha = i * 0.2
        ridge = Ridge(alpha=alpha)
        X_poly = poly.fit_transform(X_train)
        ridge.fit(X_poly, y_train)
        X_poly_val = poly.transform(X_val)
        y_pred = ridge.predict(X_poly_val)
        mse = mean_squared_error(y_val, y_pred)

        mlflow.log_param(f"alpha_{i}", alpha)
        mlflow.log_metric(f"val_mse_{i}", mse)

        if mse < best_score:
            best_score = mse
            best_model = ridge
            best_alpha = alpha

    # After loop ends
    if best_model is None:
        raise ValueError("No valid model was trained.")

    X_poly_val = poly.transform(X_val)
    y_val_pred = best_model.predict(X_poly_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_avg_delay = y_val_pred.mean()

    mlflow.log_param("best_alpha", best_alpha)
    mlflow.log_metric("val_mse_final", val_mse)
    mlflow.log_metric("val_avg_delay", val_avg_delay)

    with open("finalized_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact("finalized_model.pkl")


# In[85]:


### Step D.5: Save and Log Airport Encodings
import json

# Step 1: Build encoding dictionary with string keys and int values
airport_encoding = {
    str(k): int(v)
    for k, v in zip(le.classes_, le.transform(le.classes_))
}

# Step 2: Save to JSON file inside the 'with' block
with open("airport_encodings.json", "w") as f:
    json.dump(airport_encoding, f)

# Step 3: Log the file as an artifact in MLflow
mlflow.log_artifact("airport_encodings.json")


# In[89]:


#  D.6: Create and Log Performance Plot  Test Set
df_test_feat['DEST_AIRPORT_ENC'] = le.transform(df_test_feat['DEST_AIRPORT'])
dest_encoded_test = onehot.transform(df_test_feat[['DEST_AIRPORT_ENC']])
X_test = np.hstack((dest_encoded_test, df_test_feat[['hour_depart', 'hour_arrive']].values))
y_test = df_test_feat['DEPARTURE_DELAY'].values.reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_test_pred = best_model.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
avg_delay = np.mean(y_test_pred)


# In[91]:


# Log final metrics
mlflow.log_param("best_alpha", best_alpha)
mlflow.log_metric("test_mse", test_mse)
mlflow.log_metric("test_avg_delay", avg_delay)


# In[93]:


# Save model
with open("finalized_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
mlflow.log_artifact("finalized_model.pkl")


# In[95]:


# Save performance plot
plt.figure()
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([0, 60], [0, 60], color="red", linestyle="--")
plt.xlabel("Actual Delay")
plt.ylabel("Predicted Delay")
plt.title("Model Performance on Test Set")
plt.savefig("performance_plot.png")
mlflow.log_artifact("performance_plot.png")


# In[ ]:




