import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset you downloaded
df = pd.read_csv("Crop_recommendation.csv")

# Separate the soil/weather features from the target crop label
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a binary file
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Model trained successfully! Accuracy: {model.score(X_test, y_test)*100:.2f}%")