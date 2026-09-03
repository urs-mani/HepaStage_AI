# 🫀 HepaStage AI

### AI-Powered Liver Fibrosis Stage Prediction System

**HepaStage AI** is an AI/ML-based application designed to analyze liver-related biomarkers and predict the likely **fibrosis stage** using machine learning classification models.

The project combines **machine learning, feature engineering, biomarker validation, probability-based prediction, and history tracking** into a web-based application.

---

## 👨‍💻 Author

**Padamata Hema Mani Chandra**

---

## 📌 Project Overview

Liver fibrosis is a progressive condition in which healthy liver tissue is replaced by scar tissue. Early identification and assessment of fibrosis can support better clinical analysis.

HepaStage AI provides a machine-learning-driven approach for analyzing relevant liver biomarkers and generating a predicted fibrosis stage along with prediction probabilities.

The application is designed to provide an accessible interface for entering patient-related biomarker values, validating the input, processing the data through machine-learning models, and presenting the prediction results.

> **Note:** HepaStage AI is an educational and machine-learning project and should not be considered a substitute for professional medical diagnosis or clinical decision-making.

---

# 🎯 Project Objectives

* Validate liver biomarker inputs.
* Process healthcare-related numerical features.
* Perform feature engineering.
* Compare multiple machine-learning classification algorithms.
* Predict the fibrosis stage.
* Provide prediction probabilities.
* Maintain prediction history.
* Provide an easy-to-use web interface.
* Demonstrate the application of machine learning to healthcare analytics.

---

# 🧠 Machine Learning Pipeline

```text
Patient Biomarker Input
        ↓
Input Validation
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
Model Comparison
        ↓
Fibrosis Stage Prediction
        ↓
Prediction Probability
        ↓
History Tracking
```

---

# 🔬 Key Features

## 🧪 Biomarker Validation

The application validates liver-related biomarker inputs before sending them through the prediction pipeline.

This helps ensure that the machine-learning system receives appropriate input values.

---

## ⚙️ Feature Engineering

Interaction features are engineered from the available biomarkers to provide additional information to the classification models.

Feature engineering is an important part of the project because relationships between individual biomarkers can provide useful predictive information.

---

## 🤖 Multiple ML Classifiers

HepaStage AI compares multiple classification algorithms to evaluate their suitability for fibrosis-stage prediction.

The project evaluates **six classification models**.

---

## 📊 Fibrosis Stage Prediction

The trained machine-learning pipeline generates a predicted fibrosis stage based on the supplied biomarker information.

The application also provides prediction probabilities to communicate the model's confidence across the available prediction classes.

---

## 🗂️ Prediction History

The application maintains prediction history so previous analysis results can be reviewed.

This makes the application more useful for experimentation and analysis rather than providing only a single prediction.

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │    User Interface   │
                    │      Frontend       │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │   Backend / API     │
                    │       Flask         │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Data Processing &   │
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Machine Learning    │
                    │      Models         │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Prediction +        │
                    │ Probability Output  │
                    └─────────────────────┘
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* Classification Models
* Feature Engineering

### Backend

* Flask

### Frontend

* Web-based frontend application

### Development

* Python
* JavaScript
* HTML
* CSS

### Deployment

* Vercel

---

# 📁 Repository Structure

```text
HepaStage_AI/
│
├── docs/
│
├── frontend/
│
├── models/
│
├── src/
│
├── TODO.md
│
├── run_improved_v2.bat
│
└── README.md
```

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/urs-mani/HepaStage_AI.git
```

## 2. Open the Project

```bash
cd HepaStage_AI
```

## 3. Install Dependencies

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

> If the repository uses a different dependency setup, follow the dependency instructions included in the project files.

---

# ▶️ Running the Application

The repository includes:

```text
run_improved_v2.bat
```

For Windows, the batch file can be used to launch the improved version of the application if the required environment has been configured.

Alternatively, run the Flask application using the project's Python entry point.

---

# 🌐 Live Application

The project has a deployed web application:

🔗 **[Open HepaStage AI](https://hepa-stage-ai.vercel.app/)**

---

# 📊 Project Capabilities

| Capability           | Description                              |
| -------------------- | ---------------------------------------- |
| Biomarker Validation | Validates user-provided biomarker values |
| Feature Engineering  | Creates interaction features             |
| ML Classification    | Compares six classification models       |
| Stage Prediction     | Predicts fibrosis stage                  |
| Probability Output   | Displays prediction probabilities        |
| History Tracking     | Stores previous prediction results       |
| Web Interface        | Provides an accessible user interface    |

---

# 💡 What This Project Demonstrates

HepaStage AI demonstrates practical application of:

* Machine Learning
* Healthcare Analytics
* Python
* Scikit-learn
* Flask
* Feature Engineering
* Classification
* Probability-based prediction
* Web Application Development
* AI-assisted healthcare analysis

---

# 🔐 Responsible AI & Medical Disclaimer

HepaStage AI is a **student/academic machine-learning project** intended for educational and demonstration purposes.

The predictions generated by the system should **not be used as medical advice, diagnosis, or treatment recommendations**.

Actual fibrosis assessment should be performed by qualified healthcare professionals using appropriate clinical evaluation and validated diagnostic methods.

---

# 🔮 Future Improvements

Potential improvements include:

* Improved model validation
* Larger and more diverse datasets
* Model explainability
* Feature importance visualization
* Advanced medical-data visualizations
* Improved prediction history management
* Authentication and user accounts
* Cloud database integration
* Automated model monitoring
* Enhanced responsive UI/UX

---

# 🎓 Skills Demonstrated

This project demonstrates hands-on experience with:

**Python • Machine Learning • Scikit-learn • Flask • Feature Engineering • Classification • Healthcare Analytics • Web Development**

---

# 🔗 Links

### GitHub Repository

https://github.com/urs-mani/HepaStage_AI

### Live Application

https://hepa-stage-ai.vercel.app/

---

# 👨‍💻 Author

## Padamata Hema Mani Chandra

Final-year B.Tech Student | Aspiring Data Analyst | Developer | AI/ML Enthusiast

### Areas of Interest

* 📊 Data Analytics
* 🤖 Machine Learning
* 🧠 Artificial Intelligence
* 🌐 Web Development
* 📱 Flutter Development
* 📈 Business Intelligence
* ☁️ Cloud Computing

---

⭐ **If you find this project interesting, consider starring the repository and exploring the implementation.**
