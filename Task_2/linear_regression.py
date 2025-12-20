import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Housing.csv")   

print(data.head())    # Prints a table with the features as columns  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

X = data.drop('price', axis=1)       # Input features : .drop means takes every column except 'price' as input , along with dropping one column
y  = data['price']                   # Output : 'price' to be predicted

features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), features)], remainder='passthrough')   # Keeps the other columns unaffected
X = ct.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)        # Training the model by considering 20% of the dataset, and the rest as test-set

model = LinearRegression()
model.fit(X_train, y_train)   

y_pred = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)



#-- Plotting-----

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()


