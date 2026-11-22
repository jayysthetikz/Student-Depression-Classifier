# 🧠 Student Depression Predictor
 
An end-to-end machine learning project that predicts **depression risk in students** from demographic, academic, and lifestyle inputs, deployed as an interactive **Streamlit** web app.
 
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://student-depression-classifier-at7jrrs5gbggzm7j7nm96g.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Decision%20Tree-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
 
👉 **Try it live:** https://student-depression-classifier-at7jrrs5gbggzm7j7nm96g.streamlit.app/
 
---
 

## 🎯 Problem Statement
 
Student mental health is a growing concern. This project explores whether simple, easily collected factors (age, CGPA, living style, health condition, etc.) can be used to flag students who may be at higher risk of depression, as a **binary classification** problem.
 
## 🚀 Key Highlights
 
- Complete ML workflow: **EDA → preprocessing → model tuning → evaluation → deployment**
- Tuned `DecisionTreeClassifier` with a reported **~0.998 F1-score** during evaluation
- Interactive web app showing the **predicted class and probability** (via `predict_proba`)
- Robust **feature alignment** at inference time using saved training columns (`model_columns.pkl`)
- Deployed publicly on **Streamlit Community Cloud**
## 🛠️ Tech Stack
 
| Area | Tools |
| --- | --- |
| Language | Python |
| Data processing | pandas |
| Modeling | scikit-learn (`DecisionTreeClassifier`) |
| Web app | Streamlit |
| Model storage | Pickle (`.pkl`) |
| Experimentation | Jupyter / Google Colab |
| Deployment | Streamlit Community Cloud |
 
## 🔄 Workflow
 
1. **Data exploration:** analysed distributions and relationships in the Student Depression Dataset
2. **Preprocessing:** mapped binary/ordinal features (`Gender`, `Living_style`, `friend_circle`, `Family_background`) and one-hot encoded the remaining categoricals with `pd.get_dummies`
3. **Model training & tuning:** trained and tuned a Decision Tree classifier
4. **Evaluation:** measured performance using F1-score (add: accuracy, precision, recall, confusion matrix)
5. **Deployment:** wrapped the model in a Streamlit app with a form-based UI
6. **Inference alignment:** aligned user inputs to training columns using `model_columns.pkl` (fallback to `feature_names_in_`)
## 🧾 Input Features
 
| Feature | Input Type | Values / Range |
| --- | --- | --- |
| `Gender` | Selectbox | `Female`, `Male` |
| `Age` | Number | `15` – `40` |
| `Family_background` | Selectbox | `Lower Middle class`, `Middle class`, `Rich class` |
| `Relationship_status` | Selectbox | `Single`, `In a Relationship`, `Complicated` |
| `Marital_status` | Selectbox | `Unmarried`, `Married`, `Divorced` |
| `Living_style` | Selectbox | `Family`, `Bachelor` |
| `Current_semester` | Number | `0` – `16` |
| `CGPA` | Number | `0.00` – `4.00` |
| `friend_circle` | Selectbox | `Yes`, `No` |
| `Health_condition` | Selectbox | `Good`, `Normal`, `Abnormal`, `Critical` |
| `Cost_without_study` | Number | `0` – `100000` (monthly cost excluding study) |
| `Home_division` | Selectbox | `Dhaka`, `Chittagong`, `Rajshahi`, `Khulna`, `Barisal`, `Sylhet`, `Rangpur`, `Mymensingh` |
 
## 📁 Project Structure
 
```
.
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── depression_dt_model.pkl         # Serialized Decision Tree classifier
├── model_columns.pkl               # Saved training feature column order
├── Pipeline.ipynb                  # EDA, preprocessing, training, evaluation
├── Student_Depression_Dataset.csv  # Dataset
├── assets/                         # Screenshots and plots for this README
└── README.md
```
 
## ⚙️ Run Locally
 
```bash
# 1. Clone the repository
git clone https://github.com/jayysthetikz/Student-Depression-Classifier.git
cd Student-Depression-Classifier
 
# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
 
# 3. Install dependencies
pip install -r requirements.txt
 
# 4. Launch the app
streamlit run app.py
```
 
 
## ⚠️ Limitations & Disclaimer
 
- This is an **educational project**, **not a medical or diagnostic tool**. It must not replace professional mental health assessment.
- The dataset is limited in size and region (home divisions are from Bangladesh), so the model may not generalise to other populations.
- Predictions are based on self-reported, tabular features only.
If you or someone you know is struggling, please reach out to a qualified mental health professional or a local helpline.
 
## 🔮 Future Improvements
 
- [ ] Compare against Random Forest, Logistic Regression, and XGBoost
- [ ] Add cross-validation and a proper hold-out test report
- [ ] Add SHAP / feature-importance explanations in the app
- [ ] Handle class imbalance and calibrate probabilities
- [ ] Add unit tests and a CI workflow
## 👤 Author
 
**Jayy**: B.E. student (AI & Data Science)
 
- GitHub: [@jayysthetikz](https://github.com/jayysthetikz)
- LinkedIn: (https://www.linkedin.com/in/jayesh-yedve-425686375/)
⭐ If you found this project useful, consider giving it a star!
 
