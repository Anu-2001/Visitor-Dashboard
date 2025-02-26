import pandas as pd
from prophet import Prophet

def process_visitor_data(df, time_frame):
    """Aggregates visitor data based on the selected time frame."""
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Number of Visitors"] = df["Number of Visitors"].astype(int)

    time_formats = {
        "Daily": "Date",
        "Weekly": df["Date"].dt.strftime('%Y-%U'),
        "Monthly": df["Date"].dt.strftime('%Y-%m')
    }

    if time_frame in time_formats:
        return df.groupby(time_formats[time_frame])[["Number of Visitors"]].sum().reset_index()
    
    return df  

def process_visitor_type_data(df):
    """Prepares visitor type data for visualization."""
    df.columns = df.columns.str.strip()
    df["Month"] = df["Date"].dt.strftime('%Y-%m')
    
    return df.pivot_table(index="Month", columns="Visitor Type", 
                          values="Number of Visitors", aggfunc="sum").fillna(0).reset_index()

def predict_trends(df, selected_visitor_type, selected_weather):
    def predict_trends(df, selected_visitor_type, selected_weather):
        """Predicts visitor trends using monthly aggregated visitor type data + weather conditions."""
    
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter by visitor type
    df = df[df["Visitor Type"] == selected_visitor_type]

    # Apply weather condition filter
    if selected_weather != "All":
        filtered_df = df[df["Weather Condition"] == selected_weather]
        if len(filtered_df) > 5:  # Ensure enough data points exist
            df = filtered_df

    # Aggregate visitor numbers per month (matches stacked bar graph scale)
    monthly_visitors = df.resample("M", on="Date")["Number of Visitors"].sum().reset_index()

    if len(monthly_visitors) < 5:  # To prevent errors due to insufficient data
        return pd.DataFrame({"Date": [], "Predicted_Visitors": [], "Actual_Visitors": []})

    # Train Prophet model with seasonality for better predictions
    prophet_df = monthly_visitors.rename(columns={"Date": "ds", "Number of Visitors": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(prophet_df)

    # Extend Predictions Until Jan 2026
    future_dates = model.make_future_dataframe(periods=12, freq="M")
    future_dates = future_dates[future_dates["ds"] <= "2026-01-01"]  # Ensure predictions go until Jan 2026

    forecast = model.predict(future_dates)

    # Smooth predictions using rolling average to avoid erratic fluctuations
    forecast["yhat"] = forecast["yhat"].rolling(window=3, min_periods=1).mean()

    # Combine actual and predicted data
    result = pd.concat([
        monthly_visitors.rename(columns={"Number of Visitors": "Actual_Visitors"}).set_index("Date"),
        forecast.rename(columns={"yhat": "Predicted_Visitors"})[["ds", "Predicted_Visitors"]].set_index("ds")
    ], axis=1).reset_index().rename(columns={"index": "Date"})

    return result