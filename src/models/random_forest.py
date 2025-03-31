import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

def train_rf_model():
    # 1. Load data
    print("Loading data...")
    train_data = pd.read_csv('data/car_train_data.csv')
    test_data = pd.read_csv('data/car_test_data.csv')
    
    # 2. Handle categorical features
    print("Processing features...")
    categorical_cols = ['model', 'vehicleType', 'gearbox', 'fuelType']
    numeric_cols = ['yearOfRegistration', 'powerPS', 'kilometer', 'price']
    
    le = LabelEncoder()
    for col in categorical_cols:
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        test_data[col] = le.transform(test_data[col].astype(str))
    
    # Scale numeric features
    scaler = StandardScaler()
    train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
    test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])
    
    # 3. Prepare features and target
    X_train = train_data.drop('brand', axis=1)
    y_train = train_data['brand']
    X_test = test_data.drop('brand', axis=1)
    y_test = test_data['brand']
    
    # 4. Train Random Forest
    print("\nTraining Random Forest model...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 5. Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 6. Save model
    print("\nSaving model...")
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    # 7. Print Confusion Matrix separately
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf

if __name__ == "__main__":
    rf_model = train_rf_model()
