import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

# Load the data
data = pd.read_csv('realistic_stress_condition_with_variability.csv.csv')

# Define feature columns (X) and target columns (y)
X = data.drop(['Stress_Level', 'Stress_Type'], axis=1)
y = data[['Stress_Level', 'Stress_Type']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a RandomForestClassifier for multi-output classification
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_model)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'depression_rf_model.pkl')
print("Model saved as 'depression_rf_model.pkl'")
