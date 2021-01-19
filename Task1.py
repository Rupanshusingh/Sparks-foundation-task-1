import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

url = 'http://bit.ly/w-data'
scoreMarks = pd.read_csv(url)
print(scoreMarks.head())

X = scoreMarks.iloc[:, :-1].values
y = scoreMarks.iloc[:, 1].values
plt.scatter(X, y)
plt.xlabel("No. of hours")
plt.ylabel("Marks scored")
plt.show()



reg = LinearRegression()
reg.fit(scoreMarks[['Hours']],scoreMarks['Scores'])
print(reg.coef_)
print(reg.intercept_)
hours = 9.25
pridted_Score = reg.coef_*hours + reg.intercept_
print(pridted_Score)
