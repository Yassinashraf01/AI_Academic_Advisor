# # 🎓 AI Academic Advisor

## 📌 Project Overview

The **AI Academic Advisor** is an explainable Deep Learning web application designed to predict student academic risk levels and provide personalized academic insights.

The system analyzes academic history and behavioral features to determine whether a student is likely to:

- Graduate
- Remain Enrolled
- Drop Out

In addition to prediction, the system generates:

- Risk analysis
- Probability visualization
- Explanations for predictions
- Recommendations for improvement

The goal of the project is to assist academic advisors and students in making better educational decisions using Artificial Intelligence.

---

# 🚀 Features

✅ Student risk level prediction  
✅ Deep Learning based prediction model  
✅ Academic performance analysis  
✅ AI-generated explanations  
✅ Personalized recommendations  
✅ Probability visualization bars  
✅ Web-based interactive interface  
✅ Flask backend integration  

---

# 🧠 Machine Learning Model

The project uses a **Multi-Layer Perceptron (MLP) Neural Network** model.

### Model Architecture

- Input Layer
- Hidden Layer 1
- Hidden Layer 2
- Hidden Layer 3
- ReLU activation function
- Output Layer

The trained model predicts the academic outcome of students based on educational and behavioral data.

---

# 📊 Dataset

The project uses the following datasets:

- `dataset_697.csv`
- `student-mat.csv`

These datasets contain educational and student behavioral information such as:

- Academic grades
- Study time
- Failures
- Attendance
- Parent education
- Course difficulty
- Student background information

---

# 🛠️ Technologies Used

## Backend
- Python
- Flask

## Frontend
- HTML
- CSS
- JavaScript

## Machine Learning
- Python Notebook (Jupyter Notebook)
- Neural Networks
- MLP Classifier

---

# 📁 Project Structure

```bash
FinalProject/
│
├── data/
│   ├── output/
│   ├── dataset_697.csv
│   ├── student-mat.csv
│   └── Preprocessing.ipynb
│
├── GUI/
│   ├── app.py
│   ├── index.html
│   ├── mapping.py
│   ├── Baseline.ipynb
│   ├── NeuralNetwork.ipynb
│   ├── mlp_model.pkl
│   └── scaler.pkl
│
├── README.md
```

---

# ⚙️ How the System Works

1. The user enters student academic information in the web interface.
2. The frontend sends the data to the Flask backend using an API request.
3. `app.py` receives the request.
4. `run_academic_advisor()` from `mapping.py` is called.
5. The model predicts the student's academic outcome.
6. Additional functions generate:
   - Risk level
   - Explanations
   - Recommendations
7. Results are displayed visually in the interface.

---

# ▶️ Running the Project

## 1️⃣ Install Required Libraries

```bash
pip install flask flask-cors pandas numpy scikit-learn
```

---

## 2️⃣ Run Flask Backend

Open terminal inside the `GUI` folder and run:

```bash
python app.py
```

The Flask server will start on:

```bash
http://127.0.0.1:5000
```

---

## 3️⃣ Open Frontend

Open `index.html` in your browser.

---

# 📈 Prediction Outputs

The system predicts:

- Student Status
  - Graduate
  - Enrolled
  - Dropout

- Risk Level
  - Low Risk
  - Medium Risk
  - High Risk

The interface also displays:

- Probability bars
- AI explanations
- Personalized recommendations

---

# 💡 Example Inputs

The system evaluates features such as:

- GPA
- Approved units
- Enrolled units
- Past failures
- Study time
- Course difficulty
- Attendance behavior
- Parent education level

---

# 🔍 Explainability

One of the main goals of this project is explainability.

The system not only predicts outcomes but also explains:

- Why a student is at risk
- Which factors contributed most
- What improvements can help reduce risk

---

# 📌 Future Improvements

Possible future enhancements include:

- Real-time dashboard analytics
- More advanced deep learning architectures
- Student performance tracking over time
- Integration with university systems
- Authentication and user accounts
- Database integration
- Cloud deployment


Project Name:
**AI Academic Advisor**

---

# 📄 License

This project is for educational and academic purposes.