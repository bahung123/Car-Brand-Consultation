import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

def train_lightgbm_model():
    # Load preprocessed data
    print("Loading data...")
    train_data = pd.read_csv('data/car_train_data.csv')
    test_data = pd.read_csv('data/car_test_data.csv')
    
    # Define feature types
    print("Processing features...")
    categorical_cols = ['model', 'vehicleType', 'gearbox', 'fuelType']
    numeric_cols = ['yearOfRegistration', 'powerPS', 'kilometer', 'price']
    
    # Scale numeric features
    scaler = StandardScaler()
    train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
    test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])
    
    # Prepare data for training
    X_train = train_data.drop('brand', axis=1)
    y_train = train_data['brand']
    X_test = test_data.drop('brand', axis=1)
    y_test = test_data['brand']
    
    # Train LightGBM with adjusted hyperparameters
    print("\nTraining LightGBM model...")
    start_time = time.time()
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.01,
        min_child_samples=20,
        subsample=0.8,
        random_state=42
    )
    
    lgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = lgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save model and encoders
    print("\nSaving model and scaler...")
    model_data = {
        'model': lgb_model,
        'scaler': scaler
    }
    
    with open('models/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_lightgbm.png')
    plt.show()
    
    return lgb_model, scaler

if __name__ == "__main__":
    train_lightgbm_model()
