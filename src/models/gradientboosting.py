import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
print("Loading data...")
train_data = pd.read_csv('data/car_train_data.csv')
test_data = pd.read_csv('data/car_test_data.csv')

# 2. Process categorical variables (Assuming categorical columns are already encoded)
print("Processing features...")
categorical_cols = ['model', 'vehicleType', 'gearbox', 'fuelType']
numeric_cols = ['yearOfRegistration', 'powerPS', 'kilometer', 'price']

# 3. Prepare features and target
X_train = train_data.drop('brand', axis=1)  # Features (Without 'brand')
y_train = train_data['brand']               # Target ('brand')
X_test = test_data.drop('brand', axis=1)    # Features (Without 'brand')
y_test = test_data['brand']                 # Target ('brand')

# 4. Initialize the Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 5. Train the model
print("Training the model...")
model.fit(X_train, y_train)

# 6. Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 7. Plot Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(y_test, y_pred)

# 8. Save the model to a file
print("Saving the model to a file...")
with open('models/car_brand_classifier_gbm.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")
