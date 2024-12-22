# Building_Simple_Lineear_Regression-Model
## Simple Linear Regression
> ```pythob
> The basic steps in building a simple lenaer Regression include:
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
> It is the independent variable that is use in the predict to predict the dependent variable, in this case, salary. From this explanation, is it X_test or y_test that we use with the predict() method?
> Answer is X_test because we want to use the independent variable (X_test) to predict the y (dependent variable being salary). y_test contains the real salary, y_pred variable is predicting salary. Hence, y_pred contains the predicted salary

```python
y_pred= lr.predict(X_test)
```

#### We want to visualize the Train set. 
> + Plotting the regression line. The regression line is the predicted line as close as possible to the real salary points. And it follows a staright line, hence use the plot() function of matplotlib.pyplot
```python
plt.scatter(X_train, y_train, color= 'blue')
plt.plot(X_train, lr.predict(X_train), color= 'purple')
plt.title('Salary Vs Work Experience')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.show()
```

#### We want to visualize the Test set. 
```python
plt.scatter(X_train, y_train, color= 'blue')
plt.plot(X_train, lr.predict(X_train), color= 'purple')
plt.title('Salary Vs Work Experience')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.show()
```

