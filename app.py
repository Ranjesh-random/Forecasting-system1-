import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Demand Forecasting System", layout="wide")

# Function to generate random data
@st.cache_data
def generate_random_data(stock_codes, n_entries, start_date):
    np.random.seed(42)
    data = {
        'StockCode': np.random.choice(stock_codes, n_entries),
        'InvoiceDate': [start_date + timedelta(days=int(np.random.uniform(1, 365))) for _ in range(n_entries)],
        'Quantity': np.random.randint(1, 100, size=n_entries)
    }
    df = pd.DataFrame(data)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# Function to safely load models
def load_models():
    models = {}
    model_files = {
        'arima': "arima_model.pkl",
        'prophet': "prophet_model.pkl",
        'lstm': "lstm_model.pkl"
    }
    
    for model_name, file_name in model_files.items():
        try:
            with open(file_name, "rb") as model_file:
                models[model_name] = pickle.load(model_file)
        except FileNotFoundError:
            st.warning(f"{model_name.upper()} model file not found. This model will be unavailable.")
        except Exception as e:
            st.warning(f"Error loading {model_name.upper()} model: {str(e)}")
    
    return models

# Function to generate forecast based on model type
def generate_forecast(model, model_name, n_weeks, last_date):
    try:
        if model_name == 'arima':
            return model.forecast(steps=n_weeks)
        elif model_name == 'prophet':
            # Create future dates for Prophet
            future_dates = pd.DataFrame({
                'ds': pd.date_range(start=last_date, periods=n_weeks+1, freq='W')[1:]
            })
            forecast = model.predict(future_dates)
            return forecast['yhat'].values
        elif model_name == 'lstm':
            # Assuming the LSTM model needs the last few values for prediction
            try:
                last_sequence = model.layers[0].input_shape[1]  # Get the sequence length the model expects
            except:
                last_sequence = 4  # Default to 4 if we can't determine it
                
            last_values = weekly_data.values[-last_sequence:]
            reshaped_values = last_values.reshape((1, last_sequence, 1))
            
            predictions = []
            current_sequence = reshaped_values.copy()
            
            for _ in range(n_weeks):
                pred = model.predict(current_sequence)[0][0]
                predictions.append(pred)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0][-1][0] = pred
                
            return np.array(predictions)
    except Exception as e:
        st.warning(f"Error generating forecast for {model_name}: {str(e)}")
        return None

# Main app
def main():
    st.title("Demand Forecasting System (Random Data Version)")
    
    # User input for random data generation
    st.sidebar.header("Random Data Settings")
    n_entries = st.sidebar.slider("Number of Data Entries", min_value=100, max_value=5000, value=1000)
    n_stock_codes = st.sidebar.slider("Number of Unique Stock Codes", min_value=1, max_value=50, value=5)
    start_date = st.sidebar.date_input("Start Date for Data", value=datetime(2023, 1, 1))
    
    # Generate stock codes and data
    stock_codes = [f'ST{str(i).zfill(3)}' for i in range(1, n_stock_codes + 1)]
    data = generate_random_data(stock_codes, n_entries, start_date)
    
    # Load models
    models = load_models()
    if not models:
        st.error("No models could be loaded. Please check model files.")
        st.stop()
    
    # Sidebar for forecast settings
    st.sidebar.header("Forecast Settings")
    
    # User inputs
    selected_stock = st.sidebar.selectbox("Choose a Stock Code", options=stock_codes)
    n_weeks = st.sidebar.slider("Forecast Horizon (Weeks)", 1, 15, 4)
    
    # Filter data for selected stock and create a proper time series
    stock_data = data[data['StockCode'] == selected_stock][['InvoiceDate', 'Quantity']]
    if stock_data.empty:
        st.warning(f"No data available for stock code: {selected_stock}")
        st.stop()
    
    # Set InvoiceDate as index and sort
    stock_data = stock_data.set_index('InvoiceDate').sort_index()
    
    # Aggregate data by week
    global weekly_data  # Make it global so the generate_forecast function can access it
    weekly_data = stock_data['Quantity'].resample('W').sum()
    
    # Main content area
    st.subheader(f"Historical Data and Forecasts for Stock Code: {selected_stock}")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Historical Sales", f"{stock_data['Quantity'].sum():,.0f}")
    with col2:
        st.metric("Average Weekly Sales", f"{weekly_data.mean():,.1f}")
    with col3:
        st.metric("Weeks of Historical Data", f"{len(weekly_data)}")
    
    # Generate and plot forecasts
    for model_name, model in models.items():
        try:
            forecast = generate_forecast(model, model_name, n_weeks, weekly_data.index[-1])
            if forecast is not None:
                last_date = weekly_data.index[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(weeks=1), 
                                              periods=n_weeks, freq='W')
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(weekly_data.index, weekly_data, label="Historical Data")
                ax.plot(forecast_dates, forecast, label=f"{model_name.upper()} Forecast", 
                        marker='o', linestyle='--')
                ax.set_title(f"{model_name.upper()} Forecast for Stock {selected_stock}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Quantity")
                ax.legend()
                st.pyplot(fig)
                plt.close()
                
                # Display forecast values
                st.write(f"{model_name.upper()} Forecast Values:")
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Quantity': forecast.round(2)
                })
                st.dataframe(forecast_df)
                
        except Exception as e:
            st.error(f"Error with {model_name.upper()} model: {str(e)}")

if __name__ == "__main__":
    main()
