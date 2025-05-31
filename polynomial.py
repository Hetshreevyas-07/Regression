import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Load dataset from CSV or manually define it
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager', 'Country Manager',
                 'Region Manager', 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}
df = pd.DataFrame(data)

# Step 2: Separate independent and dependent variables
X = df[['Level']]   # Keep it as DataFrame (2D)
y = df['Salary']    # Series is okay for y

# Step 3: Transform X to polynomial features
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Step 4: Fit polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Step 5: Predict on the training set
y_pred = model.predict(X_poly)

# Step 6: Plot the results
plt.scatter(X, y, color='red')  # Original data points
plt.plot(X, y_pred, color='blue')  # Polynomial curve
plt.title('Polynomial Regression (Degree 4)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Step 7: Predict salary for Level 6.5
level = np.array([[6.5]])  # 2D array
level_poly = poly_reg.transform(level)
predicted_salary = model.predict(level_poly)
print(f"Predicted salary for level 6.5: ${predicted_salary[0]:,.2f}")

