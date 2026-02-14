
# Supply Chain Analytics with Machine Learning and Transformer Models

This repository contains an end-to-end machine learning and deep learning pipeline for **supply chain analytics**, focusing on **inventory prediction**, **cost optimization**, and **model explainability**. The framework integrates classical ML models, a Transformer-based neural network, and XAI techniques (SHAP and LIME).

---

## 📌 Project Overview

The objective of this project is to:
- Predict **inventory stock levels** using ensemble machine learning models
- Optimize **supply chain costs** using a Transformer Encoder + MLP architecture
- Provide **transparent and interpretable insights** using SHAP and LIME
- Ensure **robust preprocessing, evaluation, and reproducibility**

---

## 📂 Project Structure

```
├── supply_chain.csv          # Input dataset
├── main.py                   # Complete pipeline script
├── README.md                 # Project documentation
```

---

## 🧰 Libraries and Dependencies

### Core Libraries
- Python 3.8+
- NumPy
- Pandas

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- scikit-learn
- XGBoost

### Deep Learning
- PyTorch

### Explainable AI (XAI)
- SHAP
- LIME

Install all dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch shap lime
```

---

## 📊 Dataset Description

The dataset (`supply_chain.csv`) contains numerical and categorical attributes related to supply chain operations such as:
- Shipping times
- Shipping costs
- Stock levels
- Operational and logistics features

Missing values are handled using **forward fill**, and categorical features are encoded using **Label Encoding**.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Missing value imputation (forward fill)
- Label encoding of categorical variables
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis (EDA)
- Correlation heatmap for feature relationships
- Scatter plot for shipping time vs shipping cost

### 3. Machine Learning Models (Inventory Prediction)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

**Evaluation Metrics:**
- RMSE
- MAE
- R² Score

### 4. Deep Learning Model (Cost Optimization)
A **Transformer Encoder + MLP** architecture implemented in PyTorch:
- 2 Transformer encoder layers
- 4 attention heads
- Adam optimizer
- Mean Squared Error loss
- Trained for 50 epochs

### 5. Explainable AI
- **SHAP** for global feature importance
- **LIME** for local instance-level explanations

---

## 🧪 Model Evaluation

Models are evaluated using a **train–test split (80/20)** with:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score

This ensures fair performance comparison and generalization assessment.

---

## 🔍 Explainability & Transparency

- SHAP summary plots visualize global feature impact
- LIME explains individual predictions for interpretability
- Supports trustworthy and auditable AI decision-making

---

## ▶️ How to Run

1. Place `supply_chain.csv` in the project directory
2. Run the script:
```bash
python main.py
```
3. Visualizations and evaluation metrics will be displayed
4. SHAP and LIME explanations will be generated

---

## 📈 Applications

- Supply chain decision support systems
- Inventory optimization
- Cost forecasting and logistics planning
- Explainable AI in operations research

---

## 📄 License

This project is intended for **academic and research purposes**.

---

## ✨ Author

Developed for research and experimentation in **AI-driven supply chain analytics**.
