# 🩺 Breast Cancer Prediction using Deep Learning

This project was completed as part of my **AI/ML Internship** at **InternPe**. The final task involved developing and deploying a deep learning model to predict whether a breast tumor is **Malignant** or **Benign**, using diagnostic measurements from the Wisconsin Breast Cancer Dataset.

---

## 📌 Task Objective

Build a binary classification model using **deep learning** that takes in 30 medical input features and predicts whether a tumor is **malignant** (cancerous) or **benign** (non-cancerous).  
The final solution is wrapped in a **Streamlit-based web application** for interactive, real-time predictions.

---

## 🖥️ Web App Features

- Accepts 30 diagnostic input values  
- Predicts whether the tumor is likely **Malignant ⚠️** or **Benign 😊**  
- Clean and user-friendly interface with three-column layout  
- Includes a positive quote to reduce user anxiety  
- Fully functional locally and ready for cloud deployment  

---

## 🔗 Live App

You can try out the deployed version of this Breast Cancer Prediction App here:

👉 [Launch Live App](<your-live-link-here>)

> ⚠️ Make sure to enter all the feature values to get an accurate prediction.  
> 🔒 All predictions are generated locally in your browser—your data is never stored.

---
## 🧪 Sample Inputs to Test the Model

You can enter these manually into the Streamlit app to test predictions.

---

### 🔴 Sample 1 – Malignant

| Feature                     | Value   |
|-----------------------------|---------|
| mean radius                 | 17.99   |
| mean texture                | 10.38   |
| mean perimeter              | 122.80  |
| mean area                   | 1001.0  |
| mean smoothness             | 0.1184  |
| mean compactness            | 0.2776  |
| mean concavity              | 0.3001  |
| mean concave points         | 0.1471  |
| mean symmetry               | 0.2419  |
| mean fractal dimension      | 0.0787  |
| radius error                | 1.095   |
| texture error               | 0.9053  |
| perimeter error             | 8.589   |
| area error                  | 153.40  |
| smoothness error            | 0.0064  |
| compactness error           | 0.0490  |
| concavity error             | 0.0537  |
| concave points error        | 0.0159  |
| symmetry error              | 0.0300  |
| fractal dimension error     | 0.0062  |
| worst radius                | 25.38   |
| worst texture               | 17.33   |
| worst perimeter             | 184.60  |
| worst area                  | 2019.0  |
| worst smoothness            | 0.1622  |
| worst compactness           | 0.6656  |
| worst concavity             | 0.7119  |
| worst concave points        | 0.2654  |
| worst symmetry              | 0.4601  |
| worst fractal dimension     | 0.1189  |

---

### 🟢 Sample 2 – Benign

| Feature                     | Value   |
|-----------------------------|---------|
| mean radius                 | 12.45   |
| mean texture                | 15.70   |
| mean perimeter              | 82.57   |
| mean area                   | 477.1   |
| mean smoothness             | 0.1278  |
| mean compactness            | 0.1700  |
| mean concavity              | 0.1578  |
| mean concave points         | 0.0809  |
| mean symmetry               | 0.2087  |
| mean fractal dimension      | 0.0761  |
| radius error                | 0.3163  |
| texture error               | 1.304   |
| perimeter error             | 2.243   |
| area error                  | 24.54   |
| smoothness error            | 0.0060  |
| compactness error           | 0.0250  |
| concavity error             | 0.0311  |
| concave points error        | 0.0125  |
| symmetry error              | 0.0179  |
| fractal dimension error     | 0.0031  |
| worst radius                | 13.50   |
| worst texture               | 21.50   |
| worst perimeter             | 87.80   |
| worst area                  | 552.3   |
| worst smoothness            | 0.1534  |
| worst compactness           | 0.3564  |
| worst concavity             | 0.4115  |
| worst concave points        | 0.1532  |
| worst symmetry              | 0.3080  |
| worst fractal dimension     | 0.0833  |

---

### 🔴 Sample 3 – Malignant

| Feature                     | Value   |
|-----------------------------|---------|
| mean radius                 | 20.57   |
| mean texture                | 17.77   |
| mean perimeter              | 132.90  |
| mean area                   | 1326.0  |
| mean smoothness             | 0.08474 |
| mean compactness            | 0.1155  |
| mean concavity              | 0.1364  |
| mean concave points         | 0.0837  |
| mean symmetry               | 0.1819  |
| mean fractal dimension      | 0.0567  |
| radius error                | 0.5435  |
| texture error               | 0.7339  |
| perimeter error             | 3.398   |
| area error                  | 74.08   |
| smoothness error            | 0.0052  |
| compactness error           | 0.0138  |
| concavity error             | 0.0187  |
| concave points error        | 0.0130  |
| symmetry error              | 0.0134  |
| fractal dimension error     | 0.0035  |
| worst radius                | 24.99   |
| worst texture               | 23.41   |
| worst perimeter             | 158.80  |
| worst area                  | 1956.0  |
| worst smoothness            | 0.1238  |
| worst compactness           | 0.1866  |
| worst concavity             | 0.2416  |
| worst concave points        | 0.1860  |
| worst symmetry              | 0.2750  |
| worst fractal dimension     | 0.0890  |

✅ Paste the values into the input fields in the Streamlit app and click **Predict** to test how your model performs.

---
## 🛠️ Technologies Used

- Python  
- Streamlit  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib (for analysis)  
- Jupyter Notebook  
- H5 Format (for model saving)

---

## 📁 Project Structure
```

Breast\_Cancer\_Predictor/
│
├── BreastCancerPrediction.ipynb     # Model training, evaluation, analysis
├── model.h5                         # Trained deep learning model
├── app.py                           # Streamlit web app
├── requirements.txt                 # Required packages
├── README.md                        # Project documentation

```

---

## 🧬 Dataset Description

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains 569 instances and 30 numerical features derived from digitized images of a breast mass (e.g., radius, texture, perimeter, area, smoothness, etc.).

Key Details:
- `Diagnosis`: Target variable — **Malignant (M)** or **Benign (B)**  
- All input features are continuous and scaled before model prediction

---

## 🧠 ML Workflow

1. Data preprocessing and feature-target split  
2. Feature scaling using `StandardScaler`  
3. Deep learning model built with Keras (dense layers + sigmoid output)  
4. Training, validation, and evaluation using classification metrics  
5. Model saved as `model.h5`  
6. Developed a **Streamlit app** for live predictions  

---

## 🖥️ Web App Features

- Accepts 30 diagnostic input values  
- Predicts whether the tumor is likely **Malignant ⚠️** or **Benign 😊**  
- Clean and user-friendly interface with three-column layout  
- Includes a positive quote to reduce user anxiety  
- Fully functional locally and ready for cloud deployment  

---

## 💬 App Preview

*will share soon*
![App Screenshot](SS01.png)  
![App Screenshot](SS02.png)

---

## 💡 Future Scope

- Collect real-time diagnostic data through APIs or forms  
- Include probability/confidence score for predictions  
- Add visual graphs to show feature importance  
- Deploy the app on **Streamlit Cloud** for public access  
- Integrate with email/notification system for doctors  

---

## 🎓 Internship & Task Details

- **Internship Track**: Artificial Intelligence & Machine Learning  
- **Internship Provider**: InternPe  
- **Task Name**: Task 04 – Breast Cancer Predictor (Final Project)  
- **Environment**: Jupyter Notebook + Streamlit App (Local)

---

## 📬 Contact

**Difina George**  
📧 difina.georgecs@gmail.com  
📍 Kerala, India  
