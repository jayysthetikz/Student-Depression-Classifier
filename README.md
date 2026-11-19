# Student Depression Predictor

An interactive Streamlit web application powered by a tuned `DecisionTreeClassifier` trained on the Student Depression Dataset to estimate depression risk from student demographic, academic, and lifestyle inputs.

## Project Overview

This project takes an end-to-end machine learning workflow from notebook experimentation (`Pipeline.ipynb`) to a lightweight interactive web app (`app.py`) ready for Streamlit Community Cloud deployment.

- **Model:** Tuned `DecisionTreeClassifier` (`depression_dt_model.pkl`) achieving roughly `0.998` F1-score on evaluation.
- **Preprocessing:** Manual ordinal/binary mapping for selected categorical features (`Gender`, `Living_style`, `friend_circle`, `Family_background`), one-hot encoding (`pd.get_dummies`) for remaining categorical columns, and feature alignment (`model_columns.pkl` / `feature_names_in_`).
- **Interface:** Clean Streamlit form collecting 12 student attributes and displaying risk prediction alongside estimated probability.

## Input Features

| Feature | Type | Examples / Options |
| :--- | :--- | :--- |
| `Gender` | Categorical | `Female`, `Male` |
| `Age` | Numeric | `15` – `40` |
| `Family_background` | Categorical | `Lower Middle class`, `Middle class`, `Rich class` |
| `Relationship_status` | Categorical | `Single`, `In a Relationship`, `Complicated` |
| `Marital_status` | Categorical | `Unmarried`, `Married`, `Divorced` |
| `Living_style` | Categorical | `Family`, `Bachelor` |
| `Current_semester` | Numeric | `0` – `16` |
| `CGPA` | Numeric | `0.00` – `4.00` |
| `friend_circle` | Categorical | `Yes`, `No` |
| `Health_condition` | Categorical | `Good`, `Normal`, `Abnormal`, `Critical` |
| `Cost_without_study` | Numeric | Monthly living cost excluding study expenses |
| `Home_division` | Categorical | `Dhaka`, `Chittagong`, `Rajshahi`, `Khulna`, `Barisal`, `Sylhet`, `Rangpur`, `Mymensingh` |

## Repository Structure

```text
.
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── depression_dt_model.pkl         # Trained Decision Tree model
├── model_columns.pkl               # Saved feature column order (optional fallback)
├── Pipeline.ipynb                  # Training, evaluation, and pipeline notebook
└── Student_Depression_Dataset.csv  # Source dataset
