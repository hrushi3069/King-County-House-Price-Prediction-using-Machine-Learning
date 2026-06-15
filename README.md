# King County House Price Prediction using Machine Learning

This project predicts house sale prices in King County, Washington using supervised machine learning techniques. It is developed as part of the MM702 Data Mining and Knowledge Discovery module at the University of Brighton.

The project builds and compares multiple regression models and applies feature engineering, exploratory data analysis, and ensemble learning to improve prediction accuracy.

---

## Project Objective

The goal of this project is to:

- Predict house prices using regression models
- Identify key factors affecting house value
- Compare multiple machine learning algorithms
- Improve performance using ensemble learning techniques
- Evaluate models using standard regression metrics

---

## Dataset

The dataset contains **21,614 house sale records** from King County, USA (2014–2015).

### Features include:
- Bedrooms, bathrooms, floors
- Living area and lot size
- Grade and condition
- Waterfront and view
- Year built and renovation year
- Zipcode, latitude, longitude

### Target variable:
- House price (USD)

---

## Data Preprocessing

The following preprocessing steps were applied:

- Handled missing values using imputation
- Removed irrelevant features (e.g., ID)
- Converted date into useful time-based features
- Created new features:
  - House age
  - Renovation status
  - Years since renovation
  - Sale month and year
- Encoded categorical variables (zipcode)
- Scaled features for linear and neural models

---

## Exploratory Data Analysis

Key insights from the data:

- House prices are right-skewed with outliers
- Strong positive relationship between:
  - Living area and price
  - Grade and price
- Location (latitude, zipcode) strongly affects price
- Larger and higher-grade houses sell for higher prices

---

## Machine Learning Models

The following regression models were used:

- Linear Regression (baseline model)
- Decision Tree Regressor
- Random Forest Regressor
- Neural Network (MLP Regressor)
- Stacking Ensemble Regressor

---

## Evaluation Metrics

Models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Adjusted R² Score
- 5-Fold Cross Validation

---

## Best Model Performance

The best performing model was the **Stacking Ensemble Regressor**:

- R² Score: **0.8948**
- Adjusted R² Score: **0.8943**
- RMSE: **$125,702.79**
- MAE: **~$67,906**

The ensemble model outperformed individual regression models by combining Random Forest and Neural Network.

---

## Key Insights

- `sqft_living`, `grade`, and `location` are the strongest predictors
- Ensemble learning improves prediction accuracy
- Linear models underperform due to non-linear housing patterns
- Feature engineering significantly improves model performance

---

## Ethical Considerations

Location-based features such as zipcode, latitude, and longitude may reflect historical inequalities in housing markets. The model should be used for decision support only, not as the sole pricing system.

---

## Limitations

- Dataset is limited to King County (2014–2015)
- External factors like economy, school quality, and crime are not included
- Luxury houses show higher prediction error
- Model may not generalise to other regions or time periods

---

## Tools & Technologies
- Google Colab
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---
