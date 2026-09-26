import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 45px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-box {
        background-color: #e8f5e9;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
    }

    .prediction-price {
        font-size: 40px;
        font-weight: bold;
        color: #2e7d32;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load(
    "car_price_model.pkl"
)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("car data.csv")


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🚗 Car Price Predictor")

st.sidebar.info(
    """
    This application uses Machine Learning
    to estimate the selling price of a used car.

    Enter the car details and click
    Predict Price.
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset")

st.sidebar.write(
    f"Total Cars: **{len(df)}**"
)

st.sidebar.write(
    f"Features: **{len(df.columns)}**"
)


# ==========================================
# MAIN TITLE
# ==========================================

st.markdown(
    '<div class="main-title">🚗 Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Used Car Price Estimator'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# CAR DETAILS
# ==========================================

st.header("📝 Enter Car Details")


# First row
col1, col2, col3 = st.columns(3)


with col1:

    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2018,
        step=1
    )


with col2:

    present_price = st.number_input(
        "💵 Present Price (Lakhs)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1
    )


with col3:

    driven_kms = st.number_input(
        "🛣️ Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=30000,
        step=1000
    )


# Second row
col4, col5, col6 = st.columns(3)


with col4:

    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )


with col5:

    selling_type = st.selectbox(
        "🏪 Selling Type",
        ["Dealer", "Individual"]
    )


with col6:

    transmission = st.selectbox(
        "⚙️ Transmission",
        ["Manual", "Automatic"]
    )


# Third row
col7, col8 = st.columns(2)


with col7:

    owner = st.number_input(
        "👤 Previous Owners",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )


with col8:

    st.write("")
    st.write("")
    st.caption(
        "Enter accurate information for a more meaningful estimate."
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.markdown("---")

predict_button = st.button(
    "🔮 PREDICT CAR PRICE",
    type="primary",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    input_data = pd.DataFrame({

        "Year": [year],

        "Present_Price": [present_price],

        "Driven_kms": [driven_kms],

        "Fuel_Type": [fuel_type],

        "Selling_type": [selling_type],

        "Transmission": [transmission],

        "Owner": [owner]
    })


    # Prediction

    prediction = model.predict(
        input_data
    )[0]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.markdown(
        """
        <div class="prediction-box">

        <div style="font-size:22px;">
        💰 Estimated Selling Price
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="prediction-box">

        <div class="prediction-price">
        ₹ {prediction:.2f} Lakhs
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.success(
        "Prediction generated successfully!"
    )


# ==========================================
# ABOUT SECTION
# ==========================================

st.markdown("---")

st.header("ℹ️ About This Project")

st.write(
    """
    This project uses Machine Learning regression techniques
    to estimate the selling price of used cars.

    The model uses information such as:

    • Manufacturing Year

    • Present Price

    • Kilometers Driven

    • Fuel Type

    • Selling Type

    • Transmission

    • Number of Previous Owners

    The trained machine learning model is integrated
    with Streamlit to provide an interactive prediction
    interface.
    """
)
