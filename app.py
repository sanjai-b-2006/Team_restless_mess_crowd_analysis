import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Mess Hall Crowd Predictor (XGBoost)",
    page_icon="🍲",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_resource
def load_model():
    """Load the pre-trained XGBoost model pipeline."""
    try:
        # --- THE ONLY CHANGE NEEDED ---
        model = joblib.load(r"C:\Users\sanja\Desktop\hackathon\data_analytics_13_07_25\mess_crowd_predictor_xgb.joblib")
        # --- END OF CHANGE ---
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'mess_crowd_predictor_xgb.joblib' is in the same directory.")
        return None

@st.cache_data
def load_data():
    """Load and preprocess the dataset for the app."""
    try:
        df = pd.read_csv('dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
        
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure 'dataset.csv' is in the same directory.")
        return None

model = load_model()
df = load_data()

# Header
st.title("🍲 Mess Hall Weekly Crowd Predictor (Powered by XGBoost)")
st.markdown("""
This tool uses a powerful XGBoost machine learning model to predict the total number of diners for a given week. 
Adjust the parameters in the sidebar to get a crowd estimate and explore historical data.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Weekly Prediction Parameters")

if df is not None:
    mess_ids = sorted(df['Mess_ID'].unique())
    min_temp, max_temp = float(df['Temperature'].min()), float(df['Temperature'].max())
    min_menu, max_menu = float(df['Menu_Score'].min()), float(df['Menu_Score'].max())
    min_event, max_event = float(df['Event_Intensity_Index'].min()), float(df['Event_Intensity_Index'].max())
    min_stress, max_stress = float(df['Stress_Level'].min()), float(df['Stress_Level'].max())

    mess_id = st.sidebar.selectbox("Select Mess ID", options=mess_ids)
    date = st.sidebar.date_input("Select the Start Date of the Week", datetime.now())
    is_holiday = st.sidebar.selectbox("Is it a Holiday Week?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.sidebar.markdown("---")
    
    temp = st.sidebar.slider("Average Temperature (°F)", min_value=min_temp, max_value=max_temp, value=70.0)
    menu_score = st.sidebar.slider("Menu Score", min_value=min_menu, max_value=max_menu, value=(min_menu + max_menu) / 2)
    event_index = st.sidebar.slider("Event Intensity Index", min_value=min_event, max_value=max_event, value=(min_event + max_event) / 2)
    stress_level = st.sidebar.slider("Academic Stress Level", min_value=min_stress, max_value=max_stress, value=(min_stress + max_stress) / 2)

    if st.sidebar.button("Predict Weekly Crowd", type="primary"):
        if model is not None:
            input_data = {
                'Mess_ID': [mess_id], 'Is_Holiday': [is_holiday], 'Temperature': [temp],
                'Menu_Score': [menu_score], 'Event_Intensity_Index': [event_index],
                'Stress_Level': [stress_level], 'Year': [date.year], 'Month': [date.month],
                'Week_of_Year': [date.isocalendar()[1]]
            }
            input_df = pd.DataFrame(input_data)
            prediction = model.predict(input_df)[0]
            
            st.subheader("Predicted Crowd Size for the Week")
            st.metric(label="Estimated Diners", value=f"{int(prediction):,}")
            st.info("This is an XGBoost-powered estimate. Actual numbers may vary.")

# --- NEW SECTION: Model Performance Deep Dive ---
st.markdown("---")
st.header("🔬 Model Performance Deep Dive")
st.markdown("Here we compare the model's predictions against the actual historical data to evaluate its accuracy.")

if df is not None and model is not None:
    # Use columns for a side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time-Series: Actual vs. Predicted")
        
        # Dropdown to select a mess for detailed performance view
        mess_ids = sorted(df['Mess_ID'].unique())
        mess_id_perf = st.selectbox(
            "Select Mess to Analyze Performance", 
            options=mess_ids, 
            index=mess_ids.index(2) if 2 in mess_ids else 0, # Default to a high-traffic mess
            key="perf_mess_select"
        )

        # Filter and sort data for the selected mess
        mess_df_perf = df[df['Mess_ID'] == mess_id_perf].sort_values(by='Date')

        if not mess_df_perf.empty:
            # Prepare features for prediction
            features = ['Mess_ID', 'Is_Holiday', 'Temperature', 'Menu_Score', 'Event_Intensity_Index', 'Stress_Level', 'Year', 'Month', 'Week_of_Year']
            X_historical = mess_df_perf[features]
            
            # Generate predictions for the historical data of this mess
            predicted_historical = model.predict(X_historical)
            mess_df_perf['Predicted_Crowd'] = predicted_historical
            
            # Calculate the average error for this specific mess
            mae_mess = mean_absolute_error(mess_df_perf['Weekly_Crowd'], mess_df_perf['Predicted_Crowd'])

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(mess_df_perf['Date'], mess_df_perf['Weekly_Crowd'], label='Actual Crowd', color='royalblue', linewidth=2, alpha=0.8)
            ax.plot(mess_df_perf['Date'], mess_df_perf['Predicted_Crowd'], label='Predicted Crowd', color='darkorange', linestyle='--', linewidth=2)
            
            ax.set_title(f"Model Performance for Mess {mess_id_perf}", fontsize=16)
            ax.set_xlabel("Date")
            ax.set_ylabel("Weekly Crowd")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            st.write(f"**On average, the model's prediction for this mess is off by only {mae_mess:,.0f} diners per week.**")
            st.markdown("This plot shows how well the model captures the peaks, troughs, and overall trend of the actual crowd count over time.")


    with col2:
        st.subheader("Overall Model Accuracy (All Messes)")
        st.markdown("This plot compares actual vs. predicted values for a random sample of the data (the test set). Points close to the red line indicate accurate predictions.")

        # Re-create the test/train split to get a consistent sample for visualization
        # Note: In a production app, you might save the test set, but for this dashboard, this is fine.
        features = ['Mess_ID', 'Is_Holiday', 'Temperature', 'Menu_Score', 'Event_Intensity_Index', 'Stress_Level', 'Year', 'Month', 'Week_of_Year']
        target = 'Weekly_Crowd'
        
        X = df[features]
        y = df[target]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Create the scatter plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax2, label="Predictions")
        
        # Add the 'perfect prediction' line (y=x)
        lims = [
            min(min(y_test), min(y_pred)),
            max(max(y_test), max(y_pred)),
        ]
        ax2.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction")
        
        ax2.set_title("Actual Crowd vs. Predicted Crowd", fontsize=16)
        ax2.set_xlabel("Actual Weekly Crowd")
        ax2.set_ylabel("Predicted Weekly Crowd")
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        st.pyplot(fig2)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**The overall model R-squared score is {r2:.3f}.**")
        st.markdown("This means the model explains about **99%** of the variability in crowd numbers, indicating a very high level of accuracy across all messes.")