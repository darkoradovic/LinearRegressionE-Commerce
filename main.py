import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('Ecommerce Customers')
print(customers.head())
print(customers.info())
print(customers.describe())

sns.set_palette('GnBu_d')
sns.set_style('whitegrid')
#Created jointplot to compare the Time on Website and the Yearly Amount Spent.
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
plt.show()

#Created jointplot to compare the Time on App and the Yearly Amount Spent.
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
plt.show()

sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')
plt.show()

#Examine that kind of relationship for the whole data set.
sns.pairplot(customers)
plt.show()

#Linear plot of annual spending vs. Length of membership.
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
plt.show()

#Set a variable X equal to the numerical features of the customers and a variable Y equal to the "Yearly Amount Spent" column.
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

#Separate the data into training and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Train data
lm = LinearRegression()
lm.fit(X_train, y_train)
print('Coefficient: \n', lm.coef_)

#Predict X_test data.
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test Real Data')
plt.ylabel('Y Predictions')
plt.show()

#Calculate our model by taking the Residual Sum of Squares and the explained variance (R ^ 2).
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('MSE: ', metrics.mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#Explained variance
print('Variance: ', metrics.explained_variance_score(y_test, predictions))

#Histogram of the residuals and it should look normally distributed.
sns.distplot((y_test-predictions), bins=50)
plt.show()

coef = pd.DataFrame(lm.coef_, X.columns)
coef.columns = ['Coefficient']
print(coef)

#--------Conclusions-------
'''
1)If we keep all other values constant then an increase in Avg results. Session Length increased by 1 unit to an increase of 25.98 dollars issued.
2)If we keep all other values constant, then an increase in Time on App results in an increase of 38.59 dollars.
3)If we keep all other values constant then a rise in Time on Website by 1 unit to an increase of 0.19 dollars.
4)If all other values are kept constant then an increase in length of membership by 1 unit leads to an increase of 61.27 dollars.
'''