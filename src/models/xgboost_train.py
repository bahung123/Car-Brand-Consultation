import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    plt.savefig('xgboost_confusion_matrix.png')
    plt.close()

def train_xgb_model():
    try:
        # Load data
        print("Loading data...")
        train_data = pd.read_csv('data/car_train_data.csv')
        test_data = pd.read_csv('data/car_test_data.csv')
        
        # Handle categorical features and target
        print("Processing features...")
        categorical_cols = ['model', 'vehicleType', 'gearbox', 'fuelType']
        numeric_cols = ['yearOfRegistration', 'powerPS', 'kilometer', 'price']
        
        # Encode categorical features
        le_dict = {}
        for col in categorical_cols:
            le_dict[col] = LabelEncoder()
            train_data[col] = le_dict[col].fit_transform(train_data[col].astype(str))
            test_data[col] = le_dict[col].transform(test_data[col].astype(str))
        
        # Encode target variable
        le_brand = LabelEncoder()
        train_data['brand'] = le_brand.fit_transform(train_data['brand'])
        test_data['brand'] = le_brand.transform(test_data['brand'])
        
        # Scale numeric features
        scaler = StandardScaler()
        train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
        test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])
        
        # Prepare data
        X_train = train_data.drop('brand', axis=1)
        y_train = train_data['brand']
        X_test = test_data.drop('brand', axis=1)
        y_test = test_data['brand']
        
        n_classes = len(le_brand.classes_)
        print(f"Number of classes: {n_classes}")
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        start_time = time.time()
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=n_classes,
            random_state=42,
            n_jobs=-1
        )
        
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Save model and encoders
        print("\nSaving model and encoders...")
        model_data = {
            'model': xgb,
            'scaler': scaler,
            'accuracy': accuracy,
            'training_time': train_time,
            'label_encoders': le_dict,
            'brand_encoder': le_brand
        }
        
        with open('models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred)
        
        # Print classification report with string labels
        print("\nClassification Report:")
        class_labels = [str(label) for label in sorted(y_train.unique())]
        print(classification_report(y_test, y_pred, target_names=class_labels))

        return xgb, scaler, None  # Return None for le_brand to match expected unpacking

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    xgb_model, scaler, _ = train_xgb_model()  # Use _ to ignore third return value
