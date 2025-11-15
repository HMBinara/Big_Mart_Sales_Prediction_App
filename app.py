import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost # Required to unpickle an XGBoost model

# --- Configuration ---
MODEL_FILE = 'BMSP.pkl' 

# The model features must be in this exact order
FEATURE_COLUMNS = [
    'Item_Identifier',
    'Item_Weight',
    'Item_Fat_Content',
    'Item_Visibility',
    'Item_Type',
    'Item_MRP',
    'Outlet_Identifier',
    'Outlet_Establishment_Year',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type'
]

# Encoding mappings: friendly names → encoded values
ITEM_FAT_CONTENT_MAP = {"Low Fat": 0, "Regular": 1, "High": 2}
OUTLET_SIZE_MAP = {"Small": 0, "Medium": 1, "High": 2, "Other": 3}
OUTLET_LOCATION_MAP = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}

# --- 1. Load the Model ---
@st.cache_resource 
def load_model():
    """Load the pickled XGBoost model."""
    try:
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Sales Predictor", layout="wide")
st.title("Big Mart Sales Prediction App 🛒")
st.markdown("Predict Item Outlet Sales using the XGBoost model.")
st.divider()

if model is not None:
    # Sidebar: inputs grouped in a form
    st.sidebar.header("📋 Input Features")

    with st.sidebar.form("prediction_form"):
        # Item & price details
        st.subheader("Item Details")
        item_identifier = st.number_input("Item Identifier", min_value=0, value=100, step=1, help="Unique item code")
        item_weight = st.slider("Item Weight (kg)", min_value=1.0, max_value=50.0, value=12.85, step=0.01)
        item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.0575, step=0.0001, format="%.4f")
        item_type = st.number_input("Item Type (Code)", min_value=0, max_value=50, value=8, step=1, help="Item category code")
        item_mrp = st.slider("Item MRP (Price)", min_value=1.0, max_value=5000.0, value=150.0, step=0.1)
        item_fat_content = st.selectbox("Item Fat Content", options=list(ITEM_FAT_CONTENT_MAP.keys()), index=1)

        st.markdown("---")

        # Outlet details
        st.subheader("Outlet Details")
        outlet_identifier = st.number_input("Outlet Identifier", min_value=0, value=5, step=1, help="Unique outlet code")
        outlet_size = st.selectbox("Outlet Size", options=list(OUTLET_SIZE_MAP.keys()), index=1)
        outlet_establishment_year = st.number_input("Establishment Year", min_value=1900, max_value=2025, value=1998, step=1)
        outlet_location_type = st.selectbox("Location Tier", options=list(OUTLET_LOCATION_MAP.keys()), index=1)
        outlet_type = st.number_input("Outlet Type (Code)", min_value=0, max_value=10, value=1, step=1, help="Store type code")

        st.markdown("---")
        submitted = st.form_submit_button("💰 Predict Sales", use_container_width=True)

    # Main area: show prediction result
    st.header("📊 Prediction Result")
    st.info("Fill the inputs in the left sidebar, then click **'Predict Sales'**.")

    if submitted:
        # Convert friendly names to encoded values
        item_fat_content_encoded = ITEM_FAT_CONTENT_MAP[item_fat_content]
        outlet_size_encoded = OUTLET_SIZE_MAP[outlet_size]
        outlet_location_encoded = OUTLET_LOCATION_MAP[outlet_location_type]

        # Build input dataframe preserving column order
        input_data = pd.DataFrame([[
            item_identifier,
            item_weight,
            item_fat_content_encoded,
            item_visibility,
            item_type,
            item_mrp,
            outlet_identifier,
            outlet_establishment_year,
            outlet_size_encoded,
            outlet_location_encoded,
            outlet_type
        ]], columns=FEATURE_COLUMNS)

        try:
            prediction = model.predict(input_data)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Sales", f"${prediction:,.2f}")
            with col2:
                st.success("✅ Prediction successful!")
            
            st.balloons()
            
            # Show input summary
            st.subheader("Input Summary")
            summary_data = {
                "Item Identifier": item_identifier,
                "Item Weight (kg)": f"{item_weight:.2f}",
                "Item Fat Content": item_fat_content,
                "Item Visibility": f"{item_visibility:.4f}",
                "Item Type": item_type,
                "Item MRP": f"${item_mrp:.2f}",
                "Outlet Identifier": outlet_identifier,
                "Outlet Size": outlet_size,
                "Establishment Year": outlet_establishment_year,
                "Location Tier": outlet_location_type,
                "Outlet Type": outlet_type
            }
            st.table(pd.DataFrame(summary_data.items(), columns=["Feature", "Value"]))
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

else:
    st.info("The prediction interface is unavailable because the model could not be loaded. Please check the error message above.")