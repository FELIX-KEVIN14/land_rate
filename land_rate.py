import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection
# Collect data on land rates from various sources
data = pd.read_csv('land_rates.csv')

# Step 2: Data Preprocessing
# Preprocess the collected data to remove any inconsistencies, errors, and outliers
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data = data[data['land_size'] > 0]
data = data[data['land_rate'] > 0]
data = data[data['terrain'].isin(['hilly', 'flat', 'slope'])]

# Step 3: Exploratory Data Analysis
# Conduct exploratory data analysis to gain insights into the data and determine the most important features that influence land rates
sns.pairplot(data, x_vars=['location', 'land_size', 'terrain'], y_vars='land_rate', height=5, aspect=0.7, kind='reg')
plt.show()

# Step 4: Model Development
# Develop a machine learning model to predict land rates using several algorithms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = data[['location', 'land_size', 'terrain']]
y = data['land_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

# Step 5: Evaluation Metrics
# Use different evaluation metrics to assess the performance of the machine learning model
y_pred = gb.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Coefficient of Determination:', r2_score(y_test, y_pred))
