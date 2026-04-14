import pickle
import pandas as pd

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = pd.DataFrame([{
    'MedInc': 5.0,
    'HouseAge': 10,
    'AveRooms': 6,
    'AveBedrms': 1,
    'Population': 1000,
    'AveOccup': 3,
    'Latitude': 34,
    'Longitude': -118
}])

prediction = model.predict(sample)
print("predicted price:", prediction[0])