import pandas as pd
import os

def inverse_transorm(X):
    categorical_part = X[:, :-1]
    numeric_part = X[:, -1:]
    categorical_restored = preprocessor_Y.transformers_[0][1].inverse_transform(categorical_part)

    numeric_restored = preprocessor_Y.transformers_[1][1].inverse_transform(numeric_part)
    return np.column_stack((categorical_restored, numeric_restored))

filename = 'samples.xlsx'
filepath = f'{os.getcwd()}/{filename}'

samples_df = pd.read_excel(filepath)
samples_df



from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X = samples_df.iloc[:,1:-4]
Y = samples_df.iloc[:,-4:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Preprocessor for X
preprocessor_X = Pipeline(steps=[
    ('scaler', StandardScaler())
])

X_train_prepr = preprocessor_X.fit_transform(X_train)
X_test_prepr = preprocessor_X.transform(X_test)
# extract categories
categories = [sorted(Y['Lambda'].unique()),sorted(Y['Burst'].unique()),sorted(Y['Dz'].unique())]

# Preprocessor for y (categorical parts)
preprocessor_Y = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(categories=categories), ['Lambda', 'Burst', 'Dz']),
        ('num', StandardScaler(), ['TruePower'])
    ])

Y_train_prepr = preprocessor_Y.fit_transform(Y_train)
Y_test_prepr = preprocessor_Y.fit_transform(Y_test)



# Define and fit the MLPR model
mlpr = MLPRegressor(random_state=1, max_iter=500)
mlpr.fit(X_train_prepr, Y_train_prepr)

Y_pred = mlpr.predict(X_test_prepr)

print(inverse_transorm(Y_pred))
print(inverse_transorm(Y_test_prepr))
print(Y_test)


# Calculate metrics for the test set
mae = mean_absolute_error(Y_test_prepr, Y_pred)
mse = mean_squared_error(Y_test_prepr, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test_prepr, Y_pred)

print("\nMetrics:")
print(f"Mean Absolute Error: {mae} \nMean Squared Error: {mse} \nRoot Mean Squared Error: {rmse} \nR-squared: {r2}")


to_pred = pd.DataFrame({ "х4" : [-2.2],
                                  "х2": [3]})
X = preprocessor_X.transform(to_pred)

Y_pred = mlpr.predict(X)

print(inverse_transorm(Y_pred))