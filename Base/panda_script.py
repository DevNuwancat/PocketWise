hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [10, 20, 30, 40, 50, 60, 70, 80 ]

from sklearn.linear_model import LinearRegression
import numpy as np 

X = np.array(hours_studied).reshape(-1, 1)
y = np.array(scores)


#print(X)
#print(y)

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[11.2],[5]])
print(f"If you study 10 hours, predicted score: {prediction[0]:.0f}, {prediction[1]:.0f}")

