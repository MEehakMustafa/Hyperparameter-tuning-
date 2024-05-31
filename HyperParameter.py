import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('emails.csv')

# Drop non-numeric columns or columns that are not relevant for prediction
df = df.select_dtypes(include=['number'])

# Split the dataset into features (X) and target variable (y)
X = df.drop('spam', axis=1)  # Replace 'spam' with the name of your target column
y = df['']  # Replace 'spam' with the name of your target column

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search CV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

# Print the best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
