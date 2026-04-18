import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mlp_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TRAINING_DATA_PATH = OUTPUT_DIR / "training_data.csv"



STATUS_LABELS = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate"
}

RISK_LABELS = {
    0: "High Risk",
    1: "Medium Risk",
    2: "Low Risk"
}

FEATURE_COLUMNS = [
    "previous_grade_score",
    "background_academic_score",
    "enrolled_units_count",
    "difficulty_level",
    "past_failures_count",
    "approved_units_count",
    "study_time_level",
    "absence_rate",
    "age_group",
    "parent_education_level"
]


ENCODING_GUIDE = {
    "difficulty_level": {
        "High": 0,
        "Low": 1,
        "Medium": 2
    },
    "study_time_level": {
        "High": 0,
        "Low": 1,
        "Medium": 2
    },
    "absence_rate": {
        "Often": 0,
        "Rare": 1,
        "Sometimes": 2
    },
    "age_group": {
        "18-20": 0,
        "21-23": 1,
        "24+": 2
    }
}



def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler



def prepare_input(student_dict):
    df = pd.DataFrame([student_dict])

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df[FEATURE_COLUMNS]



def predict_student_status(student_dict, model, scaler):
    """
    Takes raw numeric input features, scales them, predicts student_status,
    and returns class probabilities too.
    """
    input_df = prepare_input(student_dict)
    scaled_input = scaler.transform(input_df)

    pred_class = int(model.predict(scaled_input)[0])
    pred_label = STATUS_LABELS[pred_class]
    probabilities = model.predict_proba(scaled_input)[0]

    return pred_class, pred_label, probabilities


def map_risk_level(student_status):
    return RISK_LABELS[student_status]


def generate_explanation(student_dict):
    reasons = []

    if student_dict["previous_grade_score"] < 80:
        reasons.append("Previous academic performance is below the expected level")

    if student_dict["background_academic_score"] < 75:
        reasons.append("Background academic score suggests weak prior preparation")

    if student_dict["approved_units_count"] < 6:
        reasons.append("Low number of approved units indicates limited academic progress")

    if student_dict["past_failures_count"] >= 2:
        reasons.append("Past failures indicate repeated academic difficulty")

    if student_dict["difficulty_level"] == 0 or student_dict["enrolled_units_count"] >= 6:
        reasons.append("Heavy academic workload may be affecting performance")

    if student_dict["study_time_level"] == 1:
        reasons.append("Low study time may reduce learning progress")

    if student_dict["absence_rate"] == 0:
        reasons.append("Frequent absences may negatively affect academic results")

    if not reasons:
        reasons.append("The profile shows generally stable academic indicators")

    return reasons


def generate_recommendation(risk_level):
    if risk_level == "Low Risk":
        return "Continue your current study plan and maintain your performance."
    elif risk_level == "Medium Risk":
        return "Increase study effort, improve attendance, and monitor your academic progress closely."
    else:
        return "Seek academic support and consider reducing workload or improving study habits."


def run_academic_advisor(student_dict):
    model, scaler = load_artifacts()

    status_num, status_label, probs = predict_student_status(student_dict, model, scaler)
    risk_level = map_risk_level(status_num)
    explanation = generate_explanation(student_dict)
    recommendation = generate_recommendation(risk_level)

    return {
        "student_status_code": status_num,
        "student_status": status_label,
        "risk_level": risk_level,
        "probabilities": {
            "Dropout": float(probs[0]),
            "Enrolled": float(probs[1]),
            "Graduate": float(probs[2])
        },
        "explanation": explanation,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    sample_student = {
        "previous_grade_score": 88,
        "background_academic_score": 85,
        "enrolled_units_count": 4,
        "difficulty_level": 1,      # Low
        "past_failures_count": 0,
        "approved_units_count": 7,
        "study_time_level": 0,      # High
        "absence_rate": 1,          # Rare
        "age_group": 0,             # 18-20
        "parent_education_level": 70
    }

    result = run_academic_advisor(sample_student)

    print("Predicted student_status:", result["student_status_code"], "-", result["student_status"])
    print("Risk level:", result["risk_level"])
    print("Probabilities:")
    print("Dropout :", round(result["probabilities"]["Dropout"], 4))
    print("Enrolled:", round(result["probabilities"]["Enrolled"], 4))
    print("Graduate:", round(result["probabilities"]["Graduate"], 4))
    print("Explanation:")
    for reason in result["explanation"]:
        print("-", reason)
    print("Recommendation:", result["recommendation"])