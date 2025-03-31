import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png')
    plt.close()

def train_svm_model():
    try:
        # Load data
        print("Loading data...")
        train_data = pd.read_csv('data/car_train_data.csv')
        test_data = pd.read_csv('data/car_test_data.csv')
        
        # Take subset for faster training
        print("Subsetting data for SVM...")
        train_sample_size = 10000  # Adjust based on memory
        train_data = train_data.sample(n=train_sample_size, random_state=42)
        test_data = test_data.sample(n=int(train_sample_size*0.2), random_state=42)
        
        # Scale features
        print("Scaling features...")
        numeric_cols = ['yearOfRegistration', 'powerPS', 'kilometer', 'price']
        scaler = StandardScaler()
        
        train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
        test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])
        
        # Prepare data
        X_train = train_data.drop('brand', axis=1)
        y_train = train_data['brand']
        X_test = test_data.drop('brand', axis=1)
        y_test = test_data['brand']
        
        # Train SVM
        print("\nTraining SVM model...")
        start_time = time.time()
        
        svm = SVC(
            kernel='rbf',
            C=1.0,  # Reduced from 10.0
            gamma='auto',
            random_state=42,
            probability=True,
            cache_size=1000  # Increase cache size
        )
        
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Display classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(y_test, y_pred)
        
        # Save model
        print("\nSaving model...")
        model_data = {
            'model': svm,
            'scaler': scaler,
            'accuracy': accuracy,
            'training_time': train_time,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        with open('models/svm_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        return svm, scaler
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    svm_model, scaler = train_svm_model()