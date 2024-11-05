import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned dataset
data = pd.read_csv('depression_assessment_dataset_filled.csv')  # Update with actual path if necessary

# Separate features (X) and target (y)
X = data.drop('Depression_Level', axis=1)
y = data['Depression_Level']

# Handle missing values in X and y
X = X.fillna(X.mean())         # Fill NaN in features with column means
y = y.fillna(y.mode()[0])      # Fill NaN in target with mode

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'depression_rf_model.pkl')

print("Model trained and saved successfully!")
