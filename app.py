from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🎓",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "depression_dt_model.pkl"


@st.cache_resource
def load_model():
  return joblib.load(MODEL_PATH)


model = load_model()

EXPECTED_COLUMNS = [
    "Gender",
    "Age",
    "Family_background",
    "Living_style",
    "Current_semester",
    "CGPA",
    "friend_circle",
    "Cost_without_study",
    "Relationship_status_Complicated",
    "Relationship_status_In a Relationship",
    "Relationship_status_Single",
    "Marital_status_Divorced",
    "Marital_status_Married",
    "Marital_status_Unmarried",
    "Health_condition_Abnormal",
    "Health_condition_Critical",
    "Health_condition_Good",
    "Health_condition_Normal",
    "Home_division_Barisal",
    "Home_division_Chittagong",
    "Home_division_Dhaka",
    "Home_division_Khulna",
    "Home_division_Mymensingh",
    "Home_division_Rajshahi",
    "Home_division_Rangpur",
    "Home_division_Sylhet",
]


def preprocess_input(raw_row: dict) -> pd.DataFrame:
  if hasattr(model, "named_steps"):
    return pd.DataFrame([raw_row])

  row = raw_row.copy()
  row["Gender"] = {"Male": 1, "Female": 0}[row["Gender"]]
  row["Living_style"] = {"Family": 1, "Bachelor": 0}[row["Living_style"]]
  row["friend_circle"] = {"Yes": 1, "No": 0}[row["friend_circle"]]
  row["Family_background"] = {
      "Lower Middle class": 0,
      "Middle class": 1,
      "Rich class": 2,
  }[row["Family_background"]]

  df = pd.DataFrame([row])
  df = pd.get_dummies(
      df,
      columns=[
          "Relationship_status",
          "Marital_status",
          "Health_condition",
          "Home_division",
      ],
      dtype=int,
  )

  feature_names = getattr(model, "feature_names_in_", None)
  target_cols = (
      list(feature_names) if feature_names is not None else EXPECTED_COLUMNS
  )
  return df.reindex(columns=target_cols, fill_value=0)


st.title("🎓 Student Depression Predictor")
st.caption("Decision Tree Classifier trained on Student Depression Dataset")

with st.form("prediction_form"):
  col1, col2 = st.columns(2)

  with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=15, max_value=40, value=21, step=1)
    family_bg = st.selectbox(
        "Family Background",
        ["Lower Middle class", "Middle class", "Rich class"],
        index=1,
    )
    relationship = st.selectbox(
        "Relationship Status",
        ["Single", "In a Relationship", "Complicated"],
    )
    marital = st.selectbox(
        "Marital Status", ["Unmarried", "Married", "Divorced"]
    )
    living = st.selectbox("Living Style", ["Family", "Bachelor"])

  with col2:
    semester = st.number_input(
        "Current Semester", min_value=0, max_value=16, value=5, step=1
    )
    cgpa = st.number_input(
        "CGPA (0.00 - 4.00)",
        min_value=0.0,
        max_value=4.0,
        value=3.45,
        step=0.01,
    )
    friend_circle = st.selectbox("Friend Circle", ["Yes", "No"])
    health = st.selectbox(
        "Health Condition", ["Good", "Normal", "Abnormal", "Critical"]
    )
    cost = st.number_input(
        "Monthly Cost Without Study",
        min_value=0,
        max_value=100000,
        value=10500,
        step=100,
    )
    division = st.selectbox(
        "Home Division",
        [
            "Dhaka",
            "Chittagong",
            "Rajshahi",
            "Khulna",
            "Barisal",
            "Sylhet",
            "Rangpur",
            "Mymensingh",
        ],
    )

  submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
  student_data = {
      "Gender": gender,
      "Age": int(age),
      "Family_background": family_bg,
      "Relationship_status": relationship,
      "Marital_status": marital,
      "Living_style": living,
      "Current_semester": int(semester),
      "CGPA": float(cgpa),
      "friend_circle": friend_circle,
      "Health_condition": health,
      "Cost_without_study": int(cost),
      "Home_division": division,
  }

  X_input = preprocess_input(student_data)
  pred = int(model.predict(X_input)[0])

  prob_text = ""
  if hasattr(model, "predict_proba"):
    prob = float(model.predict_proba(X_input)[0][1]) * 100
    prob_text = f" (Estimated Probability: **{prob:.2f}%**)"

  if pred == 1:
    st.error(f"**Prediction:** Depression Risk Detected{prob_text}")
  else:
    st.success(f"**Prediction:** No Depression Risk{prob_text}")