import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

# Sample dataset: User-item interactions
data = {
    'user_id': ['1', '2', '3', '4'],
    'item_id': ['A', 'B', 'C', 'D']
}

df = pd.DataFrame(data)

# Convert user-item matrix
user_item_matrix = pd.get_dummies(df.set_index('user_id')['item_id']).astype(int)

# Train KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_item_matrix)

# Save the model
with open("models/recommender.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
