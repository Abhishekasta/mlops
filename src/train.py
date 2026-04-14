import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os

df = pd.read_csv('data/data.csv')

x = df.drop('price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = r2_score(y_test, y_pred)
print("r2 score:", score)

if score < 0.5:
    raise Exception("model performance is too low")

os.makedirs('model', exist_ok=True)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model saved successfully")