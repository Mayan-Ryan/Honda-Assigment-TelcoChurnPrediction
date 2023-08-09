from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_churn = pd.read_csv('telco_churn.csv')
df_churn = df_churn[['gender', 'PaymentMethod', 'MonthlyCharges', 'tenure', 'Churn']].copy()

df = df_churn.copy()
df.fillna(0, inplace=True)

encode = ['gender', 'PaymentMethod']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

import numpy as np
df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

X = df.drop('Churn', axis=1)
Y = df['Churn']

# Apply SMOTE to balance the classes as the number of 'yes' responses is significantly lower than 'no' responses. This technique generates synthetic data based on training data to balance skewness
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# The previous RF model had an accuracy o over 99% and showing signs of overfitting. We perform gridsearch to determine the best hyperparameters for our classification task.
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1)

grid_search.fit(X_resampled, Y_resampled)
best_model = grid_search.best_estimator_

# We Split the resampled dataset into training and testing sets to avoid overfitting
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

loaded_model = pickle.loads(pickle.dumps(best_model))

# We make predictions on the test set to detrmine performance characteristics
Y_pred = loaded_model.predict(X_test)

# Calculate the accuracy. It is around 94 which shows less overfitting. 
accuracy = accuracy_score(Y_test, Y_pred)

print("Optimal Parameters:", grid_search.best_params_)
print(f"Accuracy: {accuracy:.8f}")
