import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Ánh xạ từ kết quả ghép train + test
VEHICLE_TYPE_MAPPING = {
    'limousine': 0, 'kombi': 1, 'bus': 2, 'kleinwagen': 3, 'suv': 4,
    'nan': 5, 'coupe': 6, 'cabrio': 7
}

FUEL_TYPE_MAPPING = {
    'diesel': 0, 'cng': 1, 'benzin': 2, 'andere': 3, 'elektro': 4,
    'hybrid': 5, 'lpg': 6, 'nan': 7  # Bổ sung dựa trên unique values [1 3 6 2 5 4 0]
}

GEARBOX_MAPPING = {
    'automatik': 0, 'manuell': 1  # Hoàn chỉnh dựa trên unique values [0 1]
}

MODEL_MAPPING = {
    'golf': 203, 'nan': 184, 'grand': 26, 'fabia': 182, '2_reihe': 53, 'andere': 133,
    '3_reihe': 49, 'passat': 23, 'navara': 10, 'ka': 158, 'polo': 164, 'twingo': 24,
    'c_max': 76, 'a_klasse': 170, 'scirocco': 209, '5er': 204, 'meriva': 174,
    'arosa': 126, 'c4': 8, 'civic': 109, 'transporter': 127, 'punto': 208, 'clio': 81,
    'kadett': 233, 'kangoo': 192, 'e_klasse': 149, 'fortwo': 65, '1er': 9, '3er': 93,
    'astra': 143, 'vito': 225, '156': 176, 'c_klasse': 161, 'forester': 156, 'a1': 166,
    'insignia': 108, 'escort': 74, 'corsa': 96, 'focus': 202, 'combo': 99, '80': 138,
    'a4': 232, 'glk': 58, '100': 38, 'z_reihe': 247, 'v40': 45, 'touran': 243,
    'megane': 69, 'a6': 175, 'sharan': 193, 'a3': 117, 'i_reihe': 90, 'zafira': 152,
    'caddy': 246, 'a2': 44, 's_klasse': 238, 'tiguan': 42, 'sl': 180, 'kaefer': 132,
    'one': 172, '1_reihe': 168, 'ibiza': 218, 'picanto': 173, '911': 0, 'leon': 131,
    'stilo': 27, '7er': 56, 'up': 1, 'm_reihe': 64, 'vectra': 212, 'scenic': 125,
    'fiesta': 154, '6_reihe': 162, 'm_klasse': 234, 'q7': 73, 'mii': 155, 'c1': 80,
    'v_klasse': 102, 'a5': 86, 'duster': 7, 'beetle': 186, 'tt': 136, 'transit': 71
}

# Ánh xạ ngược cho brand (dựa trên kết quả ghép)
BRAND_MAPPING = {
    0: 'alfa_romeo', 1: 'audi', 2: 'bmw', 3: 'chevrolet', 4: 'chrysler', 5: 'citroen',
    6: 'dacia', 7: 'daewoo', 8: 'daihatsu', 9: 'fiat', 10: 'ford', 11: 'honda',
    12: 'hyundai', 13: 'jaguar', 14: 'jeep', 15: 'kia', 16: 'lada', 17: 'lancia',
    18: 'land_rover', 19: 'mazda', 20: 'mercedes_benz', 21: 'mini', 22: 'mitsubishi',
    23: 'nissan', 24: 'opel', 25: 'peugeot', 26: 'porsche', 27: 'renault', 28: 'rover',
    29: 'saab', 30: 'seat', 31: 'skoda', 32: 'smart', 33: 'subaru', 34: 'suzuki',
    35: 'toyota', 36: 'trabant', 37: 'volkswagen', 38: 'volvo'
}

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Random Forest': 'models/random_forest_model.pkl',
        'XGBoost': 'models/xgboost_model.pkl',
        'AdaBoost': 'models/adaboost_model_improve.pkl',
        'K-Nearest Neighbors': 'models/knn_model.pkl',
        'SVM': 'models/svm_model.pkl',
        'Decision Tree': 'models/decision_tree_model.pkl',
        'Bagging': 'models/bagging_model.pkl',
        'Gradient Boosting': 'models/car_brand_classifier_gbm.pkl',
        'LightGBM': 'models/lightgbm_model.pkl',
    }
    
    for name, file in model_files.items():
        try:
            with open(file, 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, dict):
                    if 'model' not in model_data:
                        st.warning(f"{name}: Dictionary format requires 'model' key.")
                        continue
                    models[name] = model_data
                elif isinstance(model_data, (list, tuple)) and len(model_data) == 2:
                    model, scaler = model_data
                    models[name] = {'model': model, 'scaler': scaler}
                else:
                    models[name] = {'model': model_data}
                    st.info(f"{name}: Loaded as a single model without scaler.")
        except Exception as e:
            st.warning(f"Error loading {name}: {str(e)}")
            continue
    if not models:
        st.error("No models were loaded successfully. Please check model files and their paths.")
    return models

def main():
    st.title('Car Brand Prediction System')
    
    models = load_models()
    if not models:
        return
    
    model_name = st.selectbox('Select Model', list(models.keys()))
    model_dict = models[model_name]
    
    st.subheader('Enter Car Details:')
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input('Year of Registration', min_value=1900, max_value=2024, value=2020)
        model_val = st.selectbox('Model Type', options=list(MODEL_MAPPING.keys()), index=0)
        vehicle_type = st.selectbox('Vehicle Type', options=list(VEHICLE_TYPE_MAPPING.keys()), index=0)
        gearbox = st.selectbox('Gearbox', options=list(GEARBOX_MAPPING.keys()), index=0)
    
    with col2:
        power = st.number_input('Power (PS)', min_value=0, max_value=1000, value=150)
        kilometer = st.number_input('Kilometers', min_value=0, max_value=500000, value=50000)
        fuel_type = st.selectbox('Fuel Type', options=list(FUEL_TYPE_MAPPING.keys()), index=0)
        price = st.number_input('Price (€)', min_value=100, max_value=200000, value=20000)
    
    if st.button('Predict Brand'):
        if year < 1900 or power <= 0 or kilometer < 0 or price <= 0:
            st.error("Please enter valid values for all fields.")
            return
        try:
            input_data = pd.DataFrame({
                'yearOfRegistration': [year],
                'model': [MODEL_MAPPING[model_val]],
                'vehicleType': [VEHICLE_TYPE_MAPPING[vehicle_type]],
                'gearbox': [GEARBOX_MAPPING[gearbox]],
                'powerPS': [power],
                'kilometer': [kilometer],
                'fuelType': [FUEL_TYPE_MAPPING[fuel_type]],
                'price': [price]
            })
            
            if 'scaler' in model_dict and hasattr(model_dict['scaler'], 'transform'):
                numeric_cols = ['yearOfRegistration', 'powerPS', 'kilometer', 'price']
                input_data[numeric_cols] = model_dict['scaler'].transform(input_data[numeric_cols])
            
            prediction = model_dict['model'].predict(input_data)
            predicted_brand = BRAND_MAPPING.get(prediction[0], "Unknown")
            st.success(f'Predicted Brand: {predicted_brand}')
        except Exception as e:
            st.error(f'Error in prediction: {str(e)}')

if __name__ == '__main__':
    main()