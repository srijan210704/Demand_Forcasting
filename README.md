# 📊 Sales Prediction using Machine Learning

## 📝 Project Overview
The goal of this project is to build a **machine learning model** to predict the number of units sold for various products across different stores.  
Accurate sales prediction is critical for **inventory management, demand forecasting, and marketing optimization**.  

This project demonstrates the complete ML workflow — from **data preprocessing** and **feature engineering** to **model training**, **evaluation**, and **hyperparameter tuning**.

---

## 📂 Dataset
- **Input Dataset:** `train_0irEZ2H.csv`  
- **Key Columns:**
  - `week` → Original date feature (`dd/mm/yy` format).  
  - `store_id` → Identifier for the store.  
  - `sku_id` → Identifier for the product.  
  - `units_sold` → Target variable (number of units sold).  

---

## ⚙️ Steps & Methodology

### 🔹 1. Data Exploration
- Loaded dataset using **pandas**.  
- Checked structure, missing values, and distributions.  

### 🔹 2. Feature Engineering
- Extracted `day`, `month`, and `year` from the `week` column.  
- Applied **One-Hot Encoding** on `store_id` and `sku_id` to handle categorical data.  

### 🔹 3. Outlier Handling
- Identified extreme values in `units_sold` (>99th percentile).  
- Removed these outliers to improve model stability.  

### 🔹 4. Model Training
- Initial model: **RandomForestRegressor**.  
- Split dataset into **train/test** sets.  
- Evaluation metrics:
  - **R² Score**  
  - **Root Mean Squared Error (RMSE)**  

### 🔹 5. Results
- Initial RMSE: ~27  
- After preprocessing & encoding: **RMSE improved to ~17.8**  

### 🔹 6. Hyperparameter Tuning
- Used **GridSearchCV** with 5-fold Cross-Validation.  
- Tuned hyperparameters:
  - `n_estimators` (number of trees).  
  - `min_samples_split` (minimum samples to split a node).  

---

## 📈 Key Learnings
- Data preprocessing (outlier handling, encoding) can drastically improve performance.  
- RandomForest works well for tabular sales prediction tasks.  
- Hyperparameter tuning via GridSearchCV ensures a more generalized model.  

---

## 🚀 Future Enhancements
- Try **Gradient Boosting, XGBoost, or LightGBM**.  
- Explore **time-series forecasting models** (ARIMA, Prophet, LSTM).  
- Perform **feature importance analysis** to identify key drivers of sales.  
- Deploy the model using **Flask/Django + Streamlit/React** for real-time predictions.  

---

## 🛠️ Tech Stack
- **Python**  
- **Pandas, NumPy** (data manipulation)  
- **Matplotlib, Seaborn** (visualization)  
- **Scikit-learn** (ML models, GridSearchCV)  

---

## 📜 License
This project is for **educational purposes** and can be extended for real-world applications.  
