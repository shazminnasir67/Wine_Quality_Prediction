import streamlit as st
import requests
import json
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# App title and description
st.title("🍷 Wine Quality Prediction")
st.markdown("Predict wine quality based on chemical properties using Machine Learning")

# Sidebar for API configuration
st.sidebar.header("API Configuration")
api_url = st.sidebar.text_input("FastAPI URL", "http://localhost:8000")

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
    # Prepare data for API
    wine_data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }
    
    try:
        # Make API call
        with st.spinner("Making prediction..."):
            response = requests.post(f"{api_url}/predict", json=wine_data)
            
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Quality", f"{result['predicted_quality']}/10")
            
            with col2:
                st.metric("Quality Category", result['quality_category'])
            
            with col3:
                st.metric("Confidence", result['confidence'])
            
            # Quality interpretation
            st.subheader("Quality Interpretation")
            quality_score = result['predicted_quality']
            
            if quality_score <= 4:
                st.error("❌ Poor Quality Wine - Not recommended")
            elif quality_score <= 5:
                st.warning("⚠️ Fair Quality Wine - Below average")
            elif quality_score <= 6:
                st.info("✅ Good Quality Wine - Average quality")
            elif quality_score <= 7:
                st.success("🌟 Very Good Quality Wine - Above average")
            else:
                st.success("🏆 Excellent Quality Wine - Premium quality")
                
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

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
st.markdown("Built with ❤️ using Streamlit and FastAPI | Wine Quality Prediction ML Model")

# API Status Check
try:
    health_response = requests.get(f"{api_url}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Offline")

# Display API info
with st.sidebar.expander("API Information"):
    try:
        info_response = requests.get(f"{api_url}/model_info", timeout=2)
        if info_response.status_code == 200:
            api_info = info_response.json()
            st.json(api_info)
    except:
        st.write("API info unavailable")