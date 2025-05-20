import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn  # Added import for sklearn

# Load and preprocess data
crop = pd.read_csv("Crop_recommendation.csv")
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)
X = crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = crop['crop_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
ms = MinMaxScaler()
X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)

# Train model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled, y_train)

# Save model and scaler
pickle.dump(rfc, open('model.pkl', 'wb'))
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))

# Verify model accuracy
print("Model accuracy:", rfc.score(X_test_scaled, y_test))
print("NumPy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)