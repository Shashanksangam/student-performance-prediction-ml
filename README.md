# Student Performance Prediction using Machine Learning

## Overview
This project predicts student final scores using Machine Learning techniques.

The project includes:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Linear Regression
- Random Forest Regression
- Model Evaluation
- Feature Importance Visualization

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

student-performance-prediction/
│
├── data/
│   └── student_performance.csv
│
├── images/
│   ├── histogram.png
│   └── heatmap.png
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── main.py
│
├── requirements.txt
└── README.md

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run project:

```bash
python src/main.py
```

---

## Key Insights

- Test scores strongly affect final performance
- Study hours positively impact scores
- Internet overuse negatively affects performance
- Assignment completion improves academic success

---

## Future Improvements

- Add Flask Web App
- Deploy using Streamlit
- Add real student datasets
- Improve model accuracy

---

## Sample Prediction

Student 1:
Predicted Final Score = 90.83

Student 2:
Predicted Final Score = 71.42

---

## Author

Shashank Sangam