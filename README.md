# Titanic 🚢 Survival Prediction 79% (Random Forest). KAGGLE [![Kaggle](https://img.shields.io/badge/-K-20BEFF)]


A project for classifying surviving passengers of the Titanic based on popular features (gender, age, cabin class, etc.).

### Key Features:

- **Feature Engineering**: Created new binary features, such as Young (age under 30) and Large_Family (having a family of more than 2 people).
- **Correct Missing Value Imputation**: Used the median of the training data to fill in missing values for age and fare, preventing Data Leakage.
- **Validation**: Data is split into training and validation sets (80/20) to evaluate the model's real accuracy.
- **Algorithm**: Random Forest (RandomForestClassifier) with depth constraints to prevent overfitting.

### Tech Stack:

- **Python 3.13**
- **Pandas**: for working with tables.
- **Scikit-learn**: Model building, preprocessing, and metrics.

### Results:

The model shows a stable accuracy of about **81%** on validation data.

### How to use:

1. Clone the repository.
2. Make sure that the train.csv and test.csv files are in the main project folder.
3. Run the code titanik.py:
4. The final submission.csv file will be ready for upload to Kaggle.

**Goal**: Try Kaggle for the first time and improve tabular data handling.

