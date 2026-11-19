# Student Depression Predictor

An interactive Streamlit web application powered by a tuned `DecisionTreeClassifier` trained on the Student Depression Dataset to predict depression risk from student demographic, academic, and lifestyle inputs.

**Live App:** [https://student-depression-classifier-at7jrrs5gbggzm7j7nm96g.streamlit.app/](https://student-depression-classifier-at7jrrs5gbggzm7j7nm96g.streamlit.app/)

## Overview

This project moves an end-to-end machine learning workflow from a Jupyter/Colab notebook (`Pipeline.ipynb`) into a single-file Streamlit web app (`app.py`) deployed on Streamlit Community Cloud.

- **Model:** Tuned `DecisionTreeClassifier` saved as `depression_dt_model.pkl` (~0.998 F1-score during evaluation).
- **Preprocessing & Alignment:** Maps binary and ordinal features (`Gender`, `Living_style`, `friend_circle`, `Family_background`), applies one-hot encoding (`pd.get_dummies`) to categorical features, and aligns columns using `model_columns.pkl` (with fallback to `feature_names_in_` or expected feature columns).
- **Web Interface:** Interactive form built with Streamlit (`st.form`, `st.selectbox`, `st.number_input`) displaying the predicted class and estimated probability when `predict_proba` is available.

## Features

| Feature | Input Type | Values / Range |
| :--- | :--- | :--- |
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
| `Cost_without_study` | Number | `0` – `100000` (monthly cost excluding study expenses) |
| `Home_division` | Selectbox | `Dhaka`, `Chittagong`, `Rajshahi`, `Khulna`, `Barisal`, `Sylhet`, `Rangpur`, `Mymensingh` |

## Project Structure

```text
.
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python package dependencies
├── depression_dt_model.pkl         # Serialized Decision Tree classifier
├── model_columns.pkl               # Saved training feature column order
├── Pipeline.ipynb                  # EDA, preprocessing, model training, and evaluation notebook
├── Student_Depression_Dataset.csv  # Dataset used for training and evaluation
└── README.md                       # Project documentation
