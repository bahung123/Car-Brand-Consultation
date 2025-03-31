import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def process_car_data():
    # Read data
    print("\n1. LOADING DATA...")
    df = pd.read_csv('data/autos.csv')

    # Select features
    features = [
        'brand', 'yearOfRegistration', 'model', 'vehicleType', 'gearbox',
        'powerPS', 'kilometer', 'fuelType', 'price'
    ]
    df_processed = df[features].copy()

    print("\n2. DATA CLEANING...")
    # Clean price (remove outliers)
    df_processed = df_processed[df_processed['price'].between(100, 150000)]

    # Clean kilometers
    df_processed = df_processed[df_processed['kilometer'].between(0, 300000)]

    # Drop nulls
    before_clean = len(df_processed)
    df_processed = df_processed.dropna()
    after_clean = len(df_processed)

    print(f"\n3. CLEANING RESULTS:")
    print(f"Original samples: {before_clean}")
    print(f"Clean samples: {after_clean}")
    print(f"Removed samples: {before_clean - after_clean}")

    # Encode categorical columns
    print("\n4. ENCODING CATEGORICAL FEATURES...")
    label_encoders = {}
    categorical_columns = ['brand', 'model', 'vehicleType', 'gearbox', 'fuelType']

    for col in categorical_columns:
        print(f"Encoding column: {col}")
        label_encoders[col] = LabelEncoder()
        df_processed[col] = label_encoders[col].fit_transform(df_processed[col])

    # Analyze class balance
    print("\n5. CLASS DISTRIBUTION:")
    class_dist = df_processed['brand'].value_counts()
    print("\nTop 10 most common brands:")
    print(class_dist.head(10))
    print("\nBottom 10 least common brands:")
    print(class_dist.tail(10))

    # Visualize class distribution
    plt.figure(figsize=(15, 5))
    class_dist.plot(kind='bar')
    plt.title('Brand Distribution')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('brand_distribution.png')

    # Split data
    y = df_processed['brand']
    X = df_processed.drop('brand', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv('data/car_train_data.csv', index=False)
    test_data.to_csv('data/car_test_data.csv', index=False)

    print("\n6. SPLIT RESULTS:")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_car_data()
