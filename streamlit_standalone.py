import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Load model function
@st.cache_resource
def load_model():
    try:
        # Try different paths for model files
        model_paths = [
            'ml/model.pkl',           # If ml folder is in root
            'model.pkl',              # If model is in root
            './ml/model.pkl'          # Alternative path
        ]
        
        scaler_paths = [
            'ml/scaler.pkl',
            'scaler.pkl', 
            './ml/scaler.pkl'
        ]
        
        feature_paths = [
            'ml/feature_names.pkl',
            'feature_names.pkl',
            './ml/feature_names.pkl'
        ]
        
        # Load model
        model = None
        for path in model_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                break
        
        # Load scaler
        scaler = None
        for path in scaler_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    scaler = pickle.load(f)
                break
        
        # Load feature names
        feature_names = None
        for path in feature_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    feature_names = pickle.load(f)
                break
        
        if model is None or scaler is None:
            st.error("❌ Model files not found! Make sure model.pkl and scaler.pkl are in the ml/ folder.")
            return None, None, None
            
        return model, scaler, feature_names
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Load the model
model, scaler, feature_names = load_model()

# App title and description
st.title("🍷 Wine Quality Prediction")
st.markdown("Predict wine quality based on chemical properties using Machine Learning")

if model is not None:
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model not loaded. Please check your model files.")
    st.stop()

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("Wine Chemical Properties")
    
    # Input fields for wine features
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=5.0, value=0.7, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=5.0, value=0.0, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=50.0, value=1.9, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076, step=0.001)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0, step=1.0)
    
with col2:
    st.header("Additional Properties")
    
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=500.0, value=34.0, step=1.0)
    density = st.number_input("Density", min_value=0.9, max_value=1.1, value=0.9978, step=0.0001, format="%.4f")
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.51, step=0.01)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=5.0, value=0.56, step=0.01)
    alcohol = st.number_input("Alcohol (%)", min_value=0.0, max_value=20.0, value=9.4, step=0.1)

# Prediction button
if st.button("🔮 Predict Wine Quality", type="primary"):
    try:
        # Prepare features
        features = np.array([[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction = round(prediction, 1)
        
        # Determine quality category
        if prediction <= 4:
            category = "Poor"
        elif prediction <= 5:
            category = "Fair"
        elif prediction <= 6:
            category = "Good"
        elif prediction <= 7:
            category = "Very Good"
        else:
            category = "Excellent"
        
        # Simple confidence estimation
        if prediction in [3, 4, 5, 6, 7, 8]:
            confidence = "High"
        else:
            confidence = "Medium"
        
        # Display results
        st.success("Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Quality", f"{prediction}/10")
        
        with col2:
            st.metric("Quality Category", category)
        
        with col3:
            st.metric("Confidence", confidence)
        
        # Quality interpretation
        st.subheader("Quality Interpretation")
        
        if prediction <= 4:
            st.error("❌ Poor Quality Wine - Not recommended")
        elif prediction <= 5:
            st.warning("⚠️ Fair Quality Wine - Below average")
        elif prediction <= 6:
            st.info("✅ Good Quality Wine - Average quality")
        elif prediction <= 7:
            st.success("🌟 Very Good Quality Wine - Above average")
        else:
            st.success("🏆 Excellent Quality Wine - Premium quality")
            
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

# Sample data section
st.markdown("---")
st.header("📊 Sample Wine Data")

sample_wines = pd.DataFrame({
    'Wine Type': ['Red Wine Sample', 'White Wine Sample', 'Premium Red', 'Budget White'],
    'Fixed Acidity': [7.4, 6.9, 8.1, 6.2],
    'Volatile Acidity': [0.70, 0.35, 0.28, 0.45],
    'Citric Acid': [0.00, 0.34, 0.40, 0.25],
    'Residual Sugar': [1.9, 6.2, 2.1, 8.5],
    'Alcohol': [9.4, 10.1, 12.8, 9.8],
    'Expected Quality': ['5-6', '6-7', '7-8', '4-5']
})

st.dataframe(sample_wines, use_container_width=True)

# Instructions
st.markdown("---")
st.header("📝 How to Use")

st.markdown("""
1. **Enter Wine Properties**: Fill in the chemical properties of your wine sample
2. **Click Predict**: Press the prediction button to get quality score
3. **Review Results**: Check the predicted quality score (1-10 scale) and category
4. **Interpret**: Use the quality interpretation to understand wine quality

**Quality Scale:**
- 1-4: Poor Quality
- 5: Fair Quality  
- 6: Good Quality
- 7: Very Good Quality
- 8-10: Excellent Quality
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Wine Quality Prediction ML Model")

# Model info in sidebar
st.sidebar.header("🤖 Model Information")
if feature_names:
    st.sidebar.write("**Features used:**")
    for feature in feature_names:
        st.sidebar.write(f"• {feature}")



st.sidebar.write("**Model:** Random Forest Regressor")
st.sidebar.write("**Target:** Wine Quality (3-9 scale)")
st.sidebar.success("✅ Model Loaded Successfully")