# Bot Detection Project

A machine learning project to detect **bot vs human sessions** based on behavioral and interaction features. The project includes data exploration, feature engineering, model training, evaluation, and rule-based detection guidelines.

---

## 📂 Deliverables

1. **Jupyter Notebook**
   - `Bot Detection.ipynb`
   - Complete data exploration & visualization
   - Feature engineering pipeline
   - Model training and evaluation
   - Results analysis & interpretation

2. **Trained Model**
   - `bot_detection_model.pkl` (best performing model)
   - `scaler.pkl` (feature preprocessing pipeline)
   - Simple prediction function for scoring new sessions

3. **Reports**
   - `Bot Detection Report.pdf` – 1–2 pages with:
     - Key behavioral patterns
     - Model performance
     - Critical features
     - Risk scoring
     - False positive analysis  
   - `Detection Rules Summary by Ashutosh Patel.pdf` – Detection rulebook with:
     - High-risk indicators (immediate bot flags)
     - Suspicious patterns (require investigation)
     - Whitelist criteria (definitely human)

4. **Dataset**
   - `User_Sessions.csv` – Dataset used for training and evaluation

5. **Reference Document**
   - `Bot Detection Task.pdf` – Assignment/task requirements

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab
- Install required libraries:
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook "Bot Detection.ipynb"
```

### Train & Save Model
- The notebook trains models and saves:
  - `bot_detection_model.pkl`
  - `scaler.pkl`

### Make Predictions
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("bot_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example session
session = pd.DataFrame([{ "clicks": 25, "session_duration": 45, "unique_pages": 5 }])
session_scaled = scaler.transform(session)
pred = model.predict(session_scaled)
proba = model.predict_proba(session_scaled)

print(pred, proba)
```

---

## 📊 Results

- **Accuracy:** ~100%  
- **Precision:** ~100%  
- **Recall:** ~100%  
- **F1-score:** ~100%  
- **ROC AUC:** ~1.000  

---

## 🛡️ Detection Rules

- **High Risk:**  
  - Extremely short sessions with high activity  
  - Very high click rates  
  - Repeated identical sequences  

- **Suspicious:**  
  - Medium bot-score (0.5–0.8)  
  - Missing/invalid user agents  
  - Contradictory interaction signals  

- **Whitelist:**  
  - Verified human users (logged-in, MFA)  
  - Known internal/testing IP ranges  
  - Natural manual interaction traces  

---

## 📄 Project Files

```
├── Bot Detection Report.pdf
├── Bot Detection Task.pdf
├── Bot Detection.ipynb
├── Detection Rules Summary by Ashutosh Patel.pdf
├── User_Sessions.csv
├── bot_detection_model.pkl
├── scaler.pkl
├── README.md
```

---

## ✨ Future Improvements
- Try advanced models (XGBoost/LightGBM)  
- Use SHAP for feature interpretability  
- Add user-agent & IP reputation analysis  
- Build real-time scoring API (Flask/FastAPI)  

---

## 👨‍💻 Author
**Ashutosh Patel**  
GitHub: [ashu1717](https://github.com/ashu1717)  

---

## 📜 License
This project is open source and available under the **MIT License**.  
