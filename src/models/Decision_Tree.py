from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import pandas as pd

def plot_tree_visualization(model, feature_names):
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
    plt.savefig('decision_tree_visualization.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(10,6))
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title('Feature Importance')
    plt.savefig('decision_tree_importance.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Decision Tree Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('decision_tree_confusion_matrix.png')
    plt.close()

def train_decision_tree():
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
        
        # Train Decision Tree with tuned parameters
        print("\nTraining Decision Tree model...")
        start_time = time.time()
        
        dt = DecisionTreeClassifier(
            criterion='entropy',         # using 'entropy' for better splits
            max_depth=30,                # adjusted depth for more complex decision boundary
            min_samples_split=10,        # reduce overfitting by increasing min sample for split
            min_samples_leaf=5,          # make leaves more general
            max_features='sqrt',         # use the square root of the number of features
            random_state=42,
            class_weight='balanced'      # balance the class weights to address imbalanced data
        )
        
        dt.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_tree_visualization(dt, X_train.columns)
        plot_feature_importance(dt, X_train.columns)
        plot_confusion_matrix(y_test, y_pred)
        
        # Save model
        print("\nSaving model...")
        model_data = {
            'model': dt,
            'scaler': scaler,
            'accuracy': accuracy,
            'training_time': train_time,
            'feature_importance': dict(zip(X_train.columns, dt.feature_importances_))
        }
        
        with open('models/decision_tree_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        return dt, scaler
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    dt_model, scaler = train_decision_tree()
