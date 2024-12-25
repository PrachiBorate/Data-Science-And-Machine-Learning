# Car Price Prediction
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

"""Pandas : It used for data manipulation and analysis.

NumPy : It is a powerful Python library for numerical computing.

Seaborn : It provides a high-level interface for creating attractive and informative statistical graphics.

Matplotlib.pyplot : It provides a MATLAB-like interface for creating basic plots and visualizations
"""

df=pd.read_csv("Cars_info.csv")
# df.read_csv - is used to read the data from the csv files.

df
# df is the variable in which the data is getting stored from the csv files

df.head()
# head() function is used to read the starting 5 lines from the dataset

df.tail()
# tail() function is used to read the last 5 lines from the dataset

df.sample(9)
# sample(10) - function is used to get the sample of 8 rows from the given dataset

df.info()
# info() It is used to get the Information of about each and every columns

df.dtypes
# dtypes - It is used to get the datatype of the specified columns from the given datasets.

df.count()
# count() method counts the number of not empty values for each row, or column

df.shape
'''  It is used to determine the dimensions or the size of an array or DataFrame.
69 --> Rows
10--> Columns'''

df.columns
# columns - It will display the column names

df['price'].mean()
# mean() It is used to find the mean of the required column data

df['price'].median()
# median() It is used to find the mean of the required column data

df['price'].std()
#std() It is used to find the Standard Deviation of the required column data

df['price'].mode()
# mode() -  It is used to find the mode of the required column data

df.describe()
# describe() - It is used to describe the dataset into the count,mean,std,min,25%,50%(median),75%,max

df['manufacture'].value_counts()
# value_counts() --> It will count the no. of times the specfic column name present in the dataset

df["price"].unique()
 #list of unique entries

df.sort_values(["engine_s"])
# .sort_values() - It will sort the values and provides you the data in the sorting manne

df.isnull().sum()
# isnull() - It will Return u the no. of the empty cells in the row

df.drop(["engine_s","wheelbas"],axis=1,inplace=True)
# .drop() -  It is used to delete the specified columns

df.head()
# Checking for the deleted columns

df.dropna(inplace=True)
# dropna() --> It is used to delete the rows which have very less null values and then bring the data in the equal manner.

df.isnull().sum()
# There is no null cell in the rows

df.info()

"""DETECTING OUTLIER

Outlier --> It is an extreme value that falls far outside the typical range of values in a dataset.

It is a Univarient Analysis

Because we are analyzing using the single variable
"""

sns.boxplot(x=df["width"])
# There is no outlier in the width column

sns.boxplot(x=df['price'])
# There are outlier in the price  column

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
# This Calculates Interquartile Range (IQR) for each column in 25 % and 75 % and then it will substract to get the 50% of Interquartile Range (IQR)

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape

sns.boxplot(x=df['price'])
# Outliers are removed sucessfully

"""EDA (Exploratory Data Analysis)

It is used to understand the main characteristics, patterns, and insights hidden in the data.
"""

plt.scatter(df['manufacture'],df['price'])

df[df.duplicated()]

duplicate_rows_df=df[df.duplicated()]
print("Number of duplicate rows: ",duplicate_rows_df.shape)

df.count()

"""Scatter Plot :-

It is Bivarent analysis because the 2 variables that is x and y are used.
It represents data points as individual dots on a two-dimensional plane, with one variable on the x-axis and the other variable on the y-axis. Each
dot represents an observation or data point.

In this also manufacture is the x axis whereas the price  is the y axis
"""

df=df.drop_duplicates()
df.head(5)

print(df.isnull().sum())

pie=df['manufacture'].value_counts()
pie

pie.plot(kind="pie",autopct="%.2f%%")

df["fuel_cap"].value_counts().plot(kind="pie",autopct="%.2f%%")

"""Pie Chart:-

It is univarient analysis , as single variable is used.

In this manufacture and fuel_cap can show the manufacture in the pie format.
It is called a "pie chart" because the chart resembles a pie that is divided into slices, with each slice representing a particular category or data point. The size of each slice corresponds to the proportion or percentage of the whole that each category represents.
"""

sns.catplot(x ='model', hue ='fuel_cap',kind ='count', data = df)

"""Count Plot :-

In the Countplots height of each bar represents the number of occurrences of each category in the dataset. Countplots are particularly useful
for visualizing the frequency of different categories and identifying the most common or least common categories in the data.
In this the number of the fuel_cap released in each model.
"""

sns.displot(df['fuel_cap'],kde=True)

"""Displot :-
displot is used to create a histogram to visualize the distribution of a numerical variable.
It is used to create a KDE plot to visualize the estimated probability density function of a numerical variable. KDE plots show the smoothed
continuous representation of the data distribution.
"""

plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="PuOr",annot=True)
c

"""Heatmaps :-

The colors on a heatmap are used to represent the magnitude or density of the data points at different locations, with warmer colors typically indicating higher values, and cooler colors indicating lower values.

DATA PREPROCESSING
"""

x=df.iloc[:,0:5]
x

"""In x axis we r including all the columns except the targeted row which will be the y axis

"""

y=df.iloc[:,5]
y

"""In y axis we are  including the targeted row only.
Which will going to calculate and give us the value of the data.

ENCODING :-

In Encoding we will convert all the object datatype of the column to the integer for the Machine Learning Process
"""

from sklearn.preprocessing import OrdinalEncoder

"""OrdinalEncoder :-

It is used for encoding categorical features into ordinal integers

"""

oe=OrdinalEncoder()
x[["manufacture","model","sales"]]=oe.fit_transform(x[["manufacture","model","sales"]])
x

"""Fit : to perform calculations on data

Transform : apply that calculation

Converting the Object data type of columns to the integer using fit_transform with OriginalEncoder of X-axis

MODEL
"""

from sklearn.model_selection import train_test_split

"""train_test_split - It is used to split a dataset into two (or more) subsets"""

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=1)

"""x and y axis is splitted into xtrain , xtest , ytrain ,ytest
where xtrain and xtest will give the output model then the ytrain is provided to the model which gives the predicted value of y predy  which is further compared with the ytest to be accurate
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

"""LogisticRegression - It is used to perform logistic regression

classification_report - It provides a comprehensive report with metrics such as precision, recall, F1-score, and support for both classes
(positive and negative). It helps you understand the performance of the model for each class and overall accuracy.

confusion_matrix - It provides a matrix that compares the predicted class labels against the actual class labels. It helps to visualize the
number of true positives, false positives, true negatives, and false negatives, which is useful for understanding the model's performance and
identifying potential areas for improvement.

"""

#step1 -: import the model
from sklearn.linear_model import LinearRegression
#step2 -: initalize the model
linreg = LinearRegression()
#step3 -: train the model -> m & c
linreg.fit(xtrain, ytrain)
#step4 -: make prediction
ypred = linreg.predict(xtest)

"""step1 -: import the model

step2 -: initalize the model

step3 -: train the model -> m & c

step4-: make prediction
"""

from sklearn.metrics import r2_score
print(f"Accuracy : {r2_score(ytest, ypred)}")

"""Accuracy :-
 It is used to see how much accurate the data it is providing by comparing the ypred and ytest

"""

linreg.intercept_
# linreg.intercept_ :- It provides the value of the intercept

linreg.coef_
# linreg.coef_ :-  It represents the coefficients associated with each input feature in the linear regression equation.

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
ypred=model.predict(x)

"""object of linear regression class

training of model

testing of model

ypred stores the predicted value of y store hai
"""

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

"""In this We are taking the input from the user i.e manufacture, model, sales,price
And after taking input we are Predicting the value of the prices of car

SUMMARY

From this Dataset of cars_info
We have found the price of car from each manufacture
Each and Every model releases different types of the prices
By Providing the sales value of model by manufacture and price can be predicted.
"""
