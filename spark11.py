import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import streamlit as st

"""
# Simple Linear Regression Model app
Here's our app to predict the score of student based the hours they study:
"""



"""
## Table of contents
### Reading the data in
### Understanding the Data
### Data Exploration
### Simple Regression Model
 """

st.write("# Reading the data in") 
df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")

""" 
## take a look at the dataset
"""
st.write(df.head())

st.write("summarize the data")
st.write(df.describe())

st.write("""Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:""")


st.line_chart(df)
# df.plot(x='Hours', y='Scores', style='r')  
# plt.title('Hours vs Percentage')  
# plt.xlabel('Hours Studied')  
# plt.ylabel('Percentage Score')  
# plt.show()
# st.pyplot()
X = df.iloc[:, :-1].values  
y = df.iloc[:,1 ].values
fig, ax = plt.subplots()
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
ax.scatter(X, y)
st.pyplot(fig)

"""**From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

## Creating train and test dataset
"""

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


"""## **Training the Algorithm**
We have split our data into training and testing sets, and now is finally the time to train our algorithm.
"""


from sklearn.linear_model import LinearRegression  
regr = LinearRegression()  
regr.fit(X_train, y_train) 
st.write("Training complete")
print("Training complete.")


# plt.scatter(X_train, y_train,  color='blue')
# plt.plot(X_train, regr.coef_*X_train + regr.intercept_, '-r')
# plt.xlabel("Hours")
# plt.ylabel("Score")


fig, ax = plt.subplots()
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.plot(X_train, regr.coef_*X_train + regr.intercept_, '-r')
plt.ylabel('Percentage Score') 
ax.scatter(X_train, y_train,  color='blue')
st.pyplot(fig)



"""### **Making Predictions**
Now that we have trained our algorithm, it's time to make some predictions.
"""

"""# Testing data - In Hours
"""
st.write(X_test) 


"""# Predicting the scores
"""
y_pred = regr.predict(X_test) 
"""# Predicted scores
"""
st.write(y_pred)


"""# Comparing Actual vs Predicted 
"""
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
st.write(df)


# You can also test with your own data
hours = np.asanyarray(float(input()))
hours = hours.reshape(-1,1)
own_pred = regr.predict(hours)
print("No of Hours = {}".format(*hours))
print("Predicted Score = {}".format(own_pred[0]))
""" ## user input--- no of Hours
"""
st.write(*hours)

""" ## Predicted Score
"""
st.write(own_pred[0])


"""# **Evaluating the model**

The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.
"""

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))