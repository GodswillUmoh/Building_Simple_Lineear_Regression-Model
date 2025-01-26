# Building_Simple_Lineear_Regression-Model
## Simple Linear Regression
> ```pythob
> The basic steps in building a simple linaer Regression include:
> + Importing the libraries
> + Importing the dataset
> + Splitting the dataset into the Training set and Test set
> + Training the Simple Linear Regression model on the Training set
> + Predicting the Test set results
> + Visualising the Training set results
> + Visualising the Test set results
> ``` 

## Note: Regression helps to predict continuous real values such as salary but classification is when you have to predict a category or a class.
### The simple dataset for this analysis is:
|YearsOfWork |	Salary |
|------------| --------|
|1.1|	49343|
|1.3	|56205|
|1.5|	47731|
|2	|53525|
|2.2	|49891|
|2.9|	66642|
|3|	70150|
|3.2|	64445|
|3.2	|74445|
|3.7|	67189|
|3.9|	73218|
|4|	65794|
|4	|66957|
|4.1|	67081|
|4.5|	71111|
|4.9|	77938|
|5.1	|76029|
|5.3|	93088|
|5.9|	91363|
|6	|103940|
|6.8|	101738|
|7.1|	108273|
|7.9|	111302|
|8.2	|123812|
|8.7|	119431|
|9	|115582|
|9.5	|126969|
|9.6	|122635|
|10.3	|122391|
|10.5|	131872|

## The python Codes for the Linear Regression
### Importing the libraries
```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

### Importing the dataset
```python
dataset = pd.read_csv('salary_data_c.csv')
dataset.head()
```

### Splitting the dataset into the Training set and Test set
```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 3)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
```

### Training the Simple Linear Regression model on the Training set
#### The fit() method is a method of the regression that is used on the training set(X_train, y_train)
```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

#### predict is also a method of the regression use on the test set. 
> + It is the independent variable that is use to predict the dependent variable, in this case, salary. From this explanation, is it X_test or y_test that we use with the predict() method?
> + Answer is X_test because we want to use the independent variable (X_test) to predict the y (dependent variable being salary). y_test contains the real salary, y_pred variable is predicting salary. Hence, y_pred contains the predicted salary

```python
y_pred= lr.predict(X_test)
```

#### We want to visualize the Train set. 
> Plotting the regression line. The regression line is the predicted line as close as possible to the real salary points. And it follows a staright line, hence use the plot() function of matplotlib.pyplot
```python
# These are for the real observation
plt.scatter(X_train, y_train, color= 'blue')
plt.plot(X_train, lr.predict(X_train), color= 'purple')
plt.title('Salary Vs Work Experience')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.show()
```

#### We want to visualize the Test set. 
> The observation here were not trained with the model. They are more like new values. Lets see if the linear line will be close to the real values like in the training set.

```python
#These are for the real observation
plt.scatter(X_train, y_train, color= 'blue')

#We do not need to change test here as regression gives a unique line for both cases
plt.scatter(X_test, y_test, color= 'blue')
plt.plot(X_train, lr.predict(X_train), color= 'purple')
plt.title('Salary Vs Work Experience (Test Set)')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.show()
```
[To view the Graph, Click Here](https://colab.research.google.com/drive/14XbRoJjwGE-fDUezP3QtNszhLmAnA-yk#scrollTo=cbOFcIvmOIc8)

### Output:
> From the graph, Someone with 8 years experience, in the linear line, is predicted to have a salary of about 11,000. Checking the real dataset, we see that 7.9 is about 11,000, hence, the predicted value is not far from the real value.
> Also good to note that, we got a perfect linear regression line because the raw data contains values that are linear in nature. Not all dataset will be in this format. Hence, we need to also carryout for non-linear model dataset

### Predicting a single value. 
> For example, how do you predict the salary of an employee with 10 years of experience.

### Code for predicting single value
```python
print(lr.predict([[10]]))
```
### Output: from this result of 12k, the read value of 10.3 is about 12k, hence, the prediction is close
```python
[128732.9060138]
```
### Note: 
It is important to note that the predict() function takes a 2D array, this is why the 12 is inserted into two square brackets [[]].
> In summary, putting 12 into a double pair of square brackets makes the input exactly a 2D array:
> + 12→scalar
> + [12]→1D array
> + [[12]]→2D array

## Getting values for coefficient and Intercept
```python
# To get these coefficients we called the "coef_" and "intercept_" attributes from the lr object
print(lr.coef_)
print(lr.intercept_)
```
## Output: Attributes in Python are different to methods and usually return a simple value or an array of values.
```python
Coefficient [9163.72413366]
Intercept 37095.66467721284
```
> Therefore, the equation of our simple linear regression model becomes:
> __Salary=99163.72×YearsOfWork+37095.66__

