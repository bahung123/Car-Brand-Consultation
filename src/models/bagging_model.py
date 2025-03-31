import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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
    plt.title('Bagging Classifier Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('bagging_confusion_matrix.png')
    plt.close()

def train_bagging_model():
    try:
        # Load data
        print("Loading data...")
        train_data = pd.read_csv('data/car_train_data.csv')
        test_data = pd.read_csv('data/car_test_data.csv')
        
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
        
        # Create base estimator
        base_estimator = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train Bagging Classifier
        print("\nTraining Bagging model...")
        start_time = time.time()
        
        bagging = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=100,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=True,
            n_jobs=-1,
            random_state=42
        )
        
        bagging.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = bagging.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(y_test, y_pred)
        
        # Save model
        print("\nSaving model...")
        model_data = {
            'model': bagging,
            'scaler': scaler,
            'accuracy': accuracy,
            'training_time': train_time
        }
        
        with open('models/bagging_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        return bagging, scaler
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    bagging_model, scaler = train_bagging_model()