import pandas as pd
import numpy as np
from numba.cuda import profiling
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
def plot_results(y_test, y_pred, y_train, feature_importance_data):
    plt.figure(figsize=(15, 10))
    
    # Feature Importance
    plt.subplot(2, 2, 1)
    feature_imp = pd.DataFrame(feature_importance_data)
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title('Feature Importance')
    
    # Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    # Class distributions
    plt.subplot(2, 2, 3)
    pd.Series(y_train).value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Classes (Training)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    pd.Series(y_pred).value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Predicted Classes')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def train_adaboost_model():
    try:
        # Load data
        print("Loading data...")
        train_data = pd.read_csv('data/car_train_data.csv')
        test_data = pd.read_csv('data/car_test_data.csv')
        
        print("Data shapes:")
        print(f"Train data: {train_data.shape}")
        print(f"Test data: {test_data.shape}")
        
        # Scale numeric features
        print("\nProcessing features...")
        numeric_cols = ['yearOfRegistration', 'powerPS', 'kilometer', 'price']
        scaler = StandardScaler()
        train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
        test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])
        
        # Prepare data
        X_train = train_data.drop('brand', axis=1)
        y_train = train_data['brand']
        X_test = test_data.drop('brand', axis=1)
        y_test = test_data['brand']
        
        print("\nClass distribution:")
        print(pd.Series(y_train).value_counts().head())
        
        # Train AdaBoost
        print("\nTraining AdaBoost model...")
        start_time = time.time()
        
        base_dt = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        ada = AdaBoostClassifier(
            estimator=base_dt,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        ada.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = ada.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report (including F1-Score, Recall, Precision)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance data
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': ada.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot results (feature importance and class distributions)
        plt = plot_results(y_test, y_pred, y_train, feature_importance)
        plt.savefig('adaboost_results.png')
        plt.close()

        # Save model
        model_data = {
            'model': ada,
            'scaler': scaler,
            'accuracy': accuracy,
            'training_time': train_time
        }
        
        with open('models/adaboost_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Plot and save Confusion Matrix as a separate image
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('ada_confusion_matrix.png')  # Save confusion matrix as an individual image
        plt.close()

        return ada, scaler
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    ada_model, scaler = train_adaboost_model()
