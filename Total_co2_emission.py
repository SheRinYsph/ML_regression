
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import warnings

from pyparsing import results
from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from scipy import stats
#%matplotlib inline

df = pd.read_csv('Food_Production.csv')
df.head()
print(df)

print(df.isnull().any())
print(df.shape)

print(df['Total_emissions'].describe())

print(df.info())



#x=df[['Items']]
#x=df[['Land use change','Animal Feed','Farm','Processing','Transport','Packging','Retail']]
x=df[['Eutrophying emissions per kilogram (gPOâ‚„eq per kilogram)','Freshwater withdrawals per kilogram (liters per kilogram)','Land use per kilogram (mÂ² per kilogram)','Scarcity-weighted water use per kilogram (liters per kilogram)']]
y=df[['Total_emissions']]
print(x,y)
print('\n')
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
print("Training data of X",x_train,'\n', "Testing data of X",x_test,'\n', "Training data of Y",y_train, '\n',"Testing data of Y",y_test)
print('\n')
print(df.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

Lin_reg = LinearRegression()
Lin_reg.fit(x_train, y_train)
print("Intercept",Lin_reg.intercept_)
print("Slope",Lin_reg.coef_)
print(Lin_reg.score(x_test, y_test))
print('\n')

# predit value
predictions = Lin_reg.predict(x_test)
print("PREDICTED",predictions)
print("acutal values",y_test)


print("R2 Score")
print(r2_score(predictions,y_test))
# Mean Absolute Error
print("Mean Absolute Error")
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predictions)))
# mean square error
print("Mean Square Error")
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predictions)))
print("Root Mean Square Error")
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))

#explore residual
residual = y_test -predictions
print("Residual",residual)

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 2)

#plt.plot(predictions,color='red',label='Residual values total CO2 Emissions Impact')
axis[0, 0].plot(y,color='green')
axis[0, 0].set_title("total CO2 Emissions of different foods")
axis[0, 0].set_xlabel('Different food products')
axis[0, 0].set_ylabel('Y test  Total CO2 Emissions Impact')


axis[0, 1].scatter( y_test,predictions,color='blue', label='Preicted values total CO2 Emissions Impact')
axis[0, 1].scatter(y_test,y_test,color='red',label='actual values')
axis[0, 1].plot([min(x_test), max(x_test)], [min(predictions), max(predictions)], color='red') # predicted
# p1=Lin_reg.coef_*y_test[0]+Lin_reg.intercept_
# print(p1)
#axis[0, 1].plot(Lin_reg.coef_*y_test+Lin_reg.intercept_,color='red')

axis[0, 1].set_title("Predicted CO2 Emissions ")
axis[0, 1].set_ylabel('Predicted values of total CO2 Emissions Impact')
axis[0, 1].set_xlabel('Y test  Total CO2 Emissions Impact')

axis[1, 0].scatter(y_test, residual,color='violet', label='Residual values total CO2 Emissions Impact')
axis[1, 0].set_xlabel('Y test  Total CO2 Emissions Impact')
axis[1, 0].set_ylabel('Residual values of total CO2 Emissions Impact')

a=df[['Items']]
b=df[['Eutrophying emissions per kilogram (gPOâ‚„eq per kilogram)']]
c=df[['Freshwater withdrawals per kilogram (liters per kilogram)']]
d=df[['Land use per kilogram (mÂ² per kilogram)']]
e=df[['Scarcity-weighted water use per kilogram (liters per kilogram)']]

# axis[1, 1].plot(y,color='green', label='Total CO2 Emissions Impact')
# axis[1, 1].plot(b,color='pink', label='Eutrophying emissions per kilogram')
# axis[1, 1].plot(c,color='red', label='Freshwater withdrawals per kilogram')
# axis[1, 1].plot(d,color='black', label='Land use per kilogram')
# axis[1, 1].plot(e,color='blue', label='Scarcity-weighted water use per kilogram')
# axis[1, 1].set_xlabel('')
# axis[1, 1].set_ylabel('Items')


Lin_reg1 = LinearRegression()
Lin_reg1.fit(y_test, predictions)
print(Lin_reg1.intercept_)
print(Lin_reg1.coef_)
print(Lin_reg1.score(y_test, predictions))


plt.show()
