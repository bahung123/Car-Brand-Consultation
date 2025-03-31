import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('KNN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('knn_confusion_matrix.png')
    plt.close()

def train_knn_model():
    # 1. Load already encoded data
    print("Loading data...")
    train_data = pd.read_csv('data/car_train_data.csv')
    test_data = pd.read_csv('data/car_test_data.csv')
    
    # 2. Prepare features and target (no encoding needed)
    print("Processing features...")
    X_train = train_data.drop('brand', axis=1)
    y_train = train_data['brand']
    X_test = test_data.drop('brand', axis=1)
    y_test = test_data['brand']
    
    # 3. Scale numeric features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train KNN model
    print("\nTraining KNN model...")
    start_time = time.time()
    
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    
    knn.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # 5. Evaluate
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 6. Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    # 7. Save model and scaler
    print("\nSaving model and scaler...")
    with open('models/knn_model.pkl', 'wb') as f:
        pickle.dump((knn, scaler), f)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return knn, scaler

if __name__ == "__main__":
    knn_model, scaler = train_knn_model()