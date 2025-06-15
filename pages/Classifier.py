import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ➤ 1. CSS to shrink selectbox width
st.markdown("""
<style>
  .stSelectbox > div[data-baseweb="select"] > div {
    width: 140px !important;
  }
  .stSelectbox [data-testid="stVirtualDropdown"] {
    width: 140px !important;
  }
</style>
""", unsafe_allow_html=True)

st.title("🍄 Mushroom Classification App")
st.write("Select the mushroom characteristics to predict if it's edible or poisonous")

# ➤ 2. Load the dataset to get feature names and fit encoders
@st.cache_data
def load_and_prepare_data():
    """Load the dataset and prepare encoders exactly as in your notebook"""
    try:
        df = pd.read_csv("data/mushrooms.csv")  # Correct path based on your structure
        
        # Handle missing values
        if 'stalk-root' in df.columns:
            df['stalk-root'] = df['stalk-root'].replace('?', np.nan)
        
        # Drop rows with missing values
        df_cleaned = df.dropna()
        
        # Separate features and target
        y = df_cleaned['class']
        X = df_cleaned.drop('class', axis=1)
        
        return X, y, df_cleaned
    except FileNotFoundError:
        st.error("Dataset file 'data/mushrooms.csv' not found. Please check the file path.")
        return None, None, None

# ➤ 3. Load or create model and encoders
@st.cache_resource
def load_or_create_model():
    """Load existing model or create new one"""
    
    # Try to load pre-trained model first
    try:
        model = joblib.load('mushroom_model.pkl')
        ordinal_encoder = joblib.load('ordinal_encoder.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        st.success("✅ Loaded pre-trained model and encoders")
        return model, ordinal_encoder, target_encoder, feature_names
        
    except FileNotFoundError:
        st.warning("⚠️ Pre-trained model not found. Creating new model...")
        
        # Fallback: create model from scratch
        X, y, df_cleaned = load_and_prepare_data()
        
        if X is None:
            return None, None, None, None
        
        # Create encoders exactly as in your notebook
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        target_encoder = LabelEncoder()
        
        # Fit encoders
        X_encoded = ordinal_encoder.fit_transform(X)
        y_encoded = target_encoder.fit_transform(y)
        
        # Create and train model
        model = LogisticRegression(max_iter=10000)
        model.fit(X_encoded, y_encoded)
        
        st.info("🔄 Created and trained new model")
        return model, ordinal_encoder, target_encoder, X.columns.tolist()

# Load model and encoders
model, ordinal_encoder, target_encoder, feature_names = load_or_create_model()

if model is None:
    st.stop()

#Create mapping from your selectbox fields to actual dataset columns
# Based on  field definitions and typical mushroom dataset structure
field_to_column_mapping = {
    "caps_option": "cap-shape",
    "capc_option": "cap-color", 
    "bruises_option": "bruises",
    "odor_option": "odor",
    "gilla_option": "gill-attachment",
    "gills_option": "gill-size",
    "gillc_option": "gill-color",
    "sporec_option": "spore-print-color",
    "popu_option": "population",
    "habi_option": "habitat"
}

#5. Field definitions with  original options
fields = {
    "caps_option":    "Cap shape",
    "capc_option":    "Cap color", 
    "bruises_option": "Bruises?",
    "odor_option":    "Odor",
    "gilla_option":   "Gill attachment",
    "gills_option":   "Gill size",
    "gillc_option":   "Gill color",
    "sporec_option":  "Spore-print color",
    "popu_option":    "Population",
    "habi_option":    "Habitat"
}

# Get unique values for each field from the dataset
@st.cache_data
def get_field_options():
    X, _, _ = load_and_prepare_data()
    if X is None:
        return {}
    
    options = {}
    for field_key, column_name in field_to_column_mapping.items():
        if column_name in X.columns:
            options[field_key] = sorted(X[column_name].unique().tolist())
        else:
            # Fallback to your original options if column not found
            fallback_options = {
                "caps_option": ['b','c','x','f','k','s'],
                "capc_option": ['n','b','c','g','p','u','e','w','y'],
                "bruises_option": ['t','f'],
                "odor_option": ['a','l','c','y','f','m','n','p','s'],
                "gilla_option": ['a','d','f','n'],
                "gills_option": ['b','n'],
                "gillc_option": ['k','n','b','h','g','r','o','p','u','e','w','y'],
                "sporec_option": ['k','n','b','h','r','o','u','w','y'],
                "popu_option": ['a','c','n','s','v','y'],
                "habi_option": ['g','l','m','p','u','w','d']
            }
            options[field_key] = fallback_options.get(field_key, ['a', 'b'])
    
    return options

field_options = get_field_options()

#Render selectboxes in a 5×2 grid
st.write("### Select mushroom characteristics:")

keys = list(fields.keys())
user_inputs = {}

for row in range(2):
    cols = st.columns(5, gap="medium")
    for i, col in enumerate(cols):
        idx = row * 5 + i
        if idx < len(keys):
            key = keys[idx]
            label = fields[key]
            options = field_options.get(key, ['a', 'b'])
            with col:
                user_inputs[key] = st.selectbox(label, options, key=key)

# ➤ 7. Process inputs and make prediction
if st.button("🔍 Classify Mushroom", type="primary"):
    try:
        # Create input array matching your dataset structure
        input_data = []
        
        # Create a row with all features
        feature_values = {}
        
        # Map user inputs to dataset columns
        for field_key, value in user_inputs.items():
            if field_key in field_to_column_mapping:
                column_name = field_to_column_mapping[field_key]
                feature_values[column_name] = value
        
        # Create input row with all features (use first value as default for missing features)
        X, _, _ = load_and_prepare_data()
        input_row = []
        
        for column in feature_names:
            if column in feature_values:
                input_row.append(feature_values[column])
            else:
                # Use most common value for missing features
                input_row.append(X[column].mode()[0])
        
        # Convert to numpy array and reshape for encoding
        input_array = np.array(input_row).reshape(1, -1)
        
        # Encode the input using the same encoder from training
        input_encoded = ordinal_encoder.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_encoded)
        prediction_proba = model.predict_proba(input_encoded)
        
        # Decode prediction
        prediction_label = target_encoder.inverse_transform(prediction)[0]
        
        # Display results
        st.write("---")
        st.write("### 🎯 Prediction Results")
        
        # Get the highest probability class
        max_prob_idx = np.argmax(prediction_proba[0])
        max_confidence = prediction_proba[0][max_prob_idx]
        
        # Decode the prediction based on highest probability
        predicted_class = target_encoder.inverse_transform([max_prob_idx])[0]
        
        if predicted_class == 'e':  # 'e' for edible
            st.success(f"🟢 **EDIBLE** mushroom")
            st.write(f"Confidence: {max_confidence:.1%}")
        else:  # 'p' for poisonous
            st.error(f"🔴 **POISONOUS** mushroom")
            st.write(f"Confidence: {max_confidence:.1%}")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.write("**Input array (encoded):**")
            st.write(input_encoded[0])
            st.write("**Feature mapping:**")
            for field_key, value in user_inputs.items():
                if field_key in field_to_column_mapping:
                    column_name = field_to_column_mapping[field_key]
                    st.write(f"{fields[field_key]}: {value} → {column_name}")
                    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check that all fields are selected correctly.")

# ➤ 8. Display current selections
with st.expander("📋 Current Selections"):
    for key, value in user_inputs.items():
        label = fields[key]
        st.write(f"**{label}:** {value}")

# ➤ 9. Add information about the model
st.write("---")
st.write("#### ℹ️ About this Model")
st.write("""
This mushroom classifier uses a **Logistic Regression** model trained on mushroom characteristics.
The model predicts whether a mushroom is **edible** or **poisonous** based on physical features.

⚠️ **Disclaimer:** This is for educational purposes only. Never consume wild mushrooms without 
proper expert identification, as misidentification can be fatal.
""")