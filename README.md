# Diabetes Prediction Using Machine Learning

## Project Overview
This project leverages machine learning techniques to predict the likelihood of diabetes in patients based on their health parameters. The models used for classification are Logistic Regression and K-Nearest Neighbors (KNN). The user interface is developed using Streamlit for real-time prediction.

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn)
- Streamlit (for UI)
- Matplotlib/Seaborn (for data visualization)
- GridSearchCV (for hyperparameter tuning)

## Dataset
The dataset contains medical records with features like:
- Age
- Glucose Levels
- Blood Pressure
- BMI (Body Mass Index)
- Insulin Levels
- Diabetes Pedigree Function
- Outcome (0 = No Diabetes, 1 = Diabetes Present)

## Project Structure
```
Diabetes-Prediction
│
├── data
│   └── diabetes.csv
│
├── models
│   └── logistic_model.pkl
│   └── knn_model.pkl
│
├── app.py
├── preprocessing.py
├── model_training.py
├── evaluation.py
├── README.md
```

## Key Phases
1. **Data Preprocessing & EDA**
   - Handling missing values
   - Outlier detection
   - Data normalization

2. **Feature Engineering & Selection**
   - One-hot encoding for categorical data
   - SelectKBest for feature selection
   - PCA for dimensionality reduction

3. **Model Development & Training**
   - Logistic Regression
   - KNN
   - Hyperparameter tuning using GridSearchCV

4. **Model Evaluation & Optimization**
   - Confusion Matrix
   - ROC-AUC Curve
   - Precision-Recall Curve

5. **Model Deployment**
   - Streamlit web application
   - Real-time prediction

## How to Run the Project
1. Clone the repository:
```
git clone https://github.com/Anish-Khandale/Diabetes-Prediction.git
```

2. Install the required libraries:
```
pip install -r requirements.txt
```

3. Run the Streamlit application:
```
streamlit run app.py
```

## Results
- Logistic Regression performed better due to the linear nature of the data.
- KNN showed decent performance but was sensitive to data scaling.

## Future Scope
- Integration of advanced models like Random Forest or XGBoost.
- Deployment on cloud platforms.
- Adding more health parameters for better accuracy.

## Contributors
- Anish Maruti Khandale

## License
This project is licensed under the MIT License.
