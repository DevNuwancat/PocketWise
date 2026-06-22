import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# data 
data = {
    'temperature': [30, 15, 10, 28, 5, 20, 35, 12, 8, 25],
    'pizza_sold' : [20, 45, 50, 22, 60, 35, 15, 48, 55, 28]
}

# DataFrame

df = pd.DataFrame(data)
#print(df.to_string())

#print("\n Basic")
#print(df.describe())

X = df[['temperature']]
y = df['pizza_sold']

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
accuracy = r2_score(y, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

temp_to_predict = [5, 25, 35, 40]
for temp in temp_to_predict:
    input_data = pd.DataFrame([[temp]], columns=['temperature'])
    predicted_sales = model.predict(input_data)[0]
    print(f"Predicted sales for {temp} degrees Celsius is: {predicted_sales:.0f}")
          
plt.figure(figsize=(10, 6))
plt.scatter(df['temperature'], df['pizza_sold'],
            color='blue', s=100, label='Real data')
plt.plot(df['temperature'], predictions,
         color='red', label='AI prediction')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pizzas Sold')
plt.title('Pizza Shop AI Predictor 🍕')
plt.legend()
plt.show()

