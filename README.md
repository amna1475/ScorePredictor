# 🎓 Student Score Prediction

A Machine Learning project to **predict students' exam scores** based on study hours and other performance factors.  
This project applies **Linear Regression** and **Polynomial Regression** models, evaluates them with standard metrics, and visualizes predictions.

---

## 📂 Dataset
- Source: [Kaggle – Student Performance Factors](https://www.kaggle.com/)  
- Features include:
  - `weekly_self_study_hours`
  - `absence_days`
  - Subject scores: `math_score`, `english_score`, `physics_score`, `chemistry_score`, `history_score`, `biology_score`, `geography_score`

A new feature `ExamScore` was created by averaging all subject scores.

---

## 🛠️ Tools & Libraries
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🔍 Project Workflow

### 1. Data Preprocessing
- Loaded dataset from Kaggle  
- Handled missing values  
- Created **ExamScore** as the average of all subject scores  

### 2. Exploratory Data Analysis (EDA)
- Visualized study hours, absences, and subject scores with **pairplots**  
- Checked distributions and feature correlations  

### 3. Model Training
- **Linear Regression** to predict exam scores from study hours  
- **Polynomial Regression** to capture non-linear trends  

### 4. Model Evaluation
- **MAE (Mean Absolute Error)**  
- **MSE (Mean Squared Error)**  
- **RMSE (Root Mean Squared Error)**  
- **R² Score**  

### 5. Visualization
- Scatter plot of actual vs predicted scores  
- Regression line for linear model  
- Polynomial regression curve  

---

## 📊 Results
- Linear Regression provided a good baseline.  
- Polynomial Regression improved accuracy by capturing non-linear patterns.  
- Visualizations confirmed a positive relationship between **study hours** and **exam scores**.

---

✨ Author

👩‍💻 Amna Bibi

Email: amna.sparish@gmail.com

Linkedin: https://www.linkedin.com/in/amna-bibi-5a82a62b3/

Feel free to give feedback, I will be happy to learn and improve!
