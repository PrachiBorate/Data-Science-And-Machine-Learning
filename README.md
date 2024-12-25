# Car_Price_Prediction
This project involves analyzing car data to predict car prices based on features like manufacturer, model, sales, and other specifications. The workflow includes data cleaning, exploratory data analysis (EDA), preprocessing, and predictive modeling using Linear Regression.

## Features
- Data Analysis: Computes statistics (mean, median, mode) and detects outliers.
- Visualization: Scatter plots, pie charts, and heatmaps to explore relationships and trends.
- Machine Learning: Predicts car prices using regression models

## Technologies Used
- Pandas, NumPy: For data manipulation and numerical computing.
- Matplotlib, Seaborn: For data visualization.
- Scikit-learn: For machine learning (Linear Regression, Logistic Regression).

## Setup Instructions
### Prerequisites
- Python 3.x
- pip (Python package installer)

## Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

## Code


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Cars_info.csv")

df

df.head()

df.tail()

df.sample(9)

df.info()

df.dtypes

df.count()

df.shape

df.columns

df['price'].mean()

df['price'].median()

df['price'].std()

df['price'].mode()

df.describe()

df['manufacture'].value_counts()

df["price"].unique()

df.sort_values(["engine_s"])

df.isnull().sum()

df.drop(["engine_s","wheelbas"],axis=1,inplace=True)

df.head()

df.dropna(inplace=True)

df.isnull().sum()

df.info()

sns.boxplot(x=df["width"])

sns.boxplot(x=df['price'])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape

sns.boxplot(x=df['price'])

plt.scatter(df['manufacture'],df['price'])

df[df.duplicated()]

duplicate_rows_df=df[df.duplicated()]
print("Number of duplicate rows: ",duplicate_rows_df.shape)

df.count()

df=df.drop_duplicates()
df.head(5)

print(df.isnull().sum())

pie=df['manufacture'].value_counts()
pie

pie.plot(kind="pie",autopct="%.2f%%")

df["fuel_cap"].value_counts().plot(kind="pie",autopct="%.2f%%")

sns.catplot(x ='model', hue ='fuel_cap',kind ='count', data = df)

sns.displot(df['fuel_cap'],kde=True)

plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="PuOr",annot=True)
c

x=df.iloc[:,0:5]
x

y=df.iloc[:,5]
y

from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder()
x[["manufacture","model","sales"]]=oe.fit_transform(x[["manufacture","model","sales"]])
x

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(xtrain, ytrain)
ypred = linreg.predict(xtest)

from sklearn.metrics import r2_score
print(f"Accuracy : {r2_score(ytest, ypred)}")

linreg.intercept_

linreg.coef_

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
ypred=model.predict(x)


ypred

from sklearn.metrics import r2_score
r2_score(y,ypred)

model.intercept_

model.coef_

x

r=int(input("\n 0.Acura \t 1.Audi \t 2.BMW\t 3.Honda \n enter the manufacture: "))

a=int(input("\n0.Integral  1.A8, 2. 328i \t 3.Passport\n enter the model: "))

g=int(input("\n 0.16.919\t 1.38 \t 9.231 \t 12.855\n enter the sales: "))

m=0.0

year=int(input("\nEnter the prices : "))

unknown_y=[[r,a,g,m,year]]

ynew=model.predict(unknown_y)

print("\n\nThe prices of car: ",ynew)

