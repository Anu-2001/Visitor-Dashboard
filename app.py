import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from process_data import process_visitor_data, process_visitor_type_data, predict_trends

# Set page title
st.title("Draco National Park Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove any trailing spaces from column names

    # Visitor Analysis (Daily, Weekly, Monthly)
    st.header("Visitor Analysis")
    time_option = st.selectbox("Select Time Frame", ["Daily", "Weekly", "Monthly"])
    visitor_data = process_visitor_data(df, time_option)
    fig_visitor = px.line(visitor_data, 
                          x=visitor_data.columns[0],  # Automatically selects 'Date', 'Week', or 'Month'
                          y="Number of Visitors", 
                          title="Visitor Trends",
                          line_shape="linear",
                          color_discrete_sequence=["darkblue"])  # Set color to darkblue
    st.plotly_chart(fig_visitor)

    # Visitor Type Analysis (Stacked Bar Chart with Total Count on Top)
    st.header("Visitor Type Analysis")
    visitor_type_data = process_visitor_type_data(df)
    visitor_type_data["Total Visitors"] = visitor_type_data.iloc[:, 1:].sum(axis=1)
    visitor_type_data["Month"] = pd.to_datetime(visitor_type_data["Month"]).dt.strftime('%b')  # Format month name only

    fig_type = px.bar(visitor_type_data, 
                      x="Month", 
                      y=visitor_type_data.columns[1:-1],  # Dynamically selects all visitor type columns
                      title="Visitor Type Counts", 
                      barmode="stack",
                      text_auto=True)  # Adds text labels inside bars
    
    # Add total visitor count as annotations slightly above each bar
    for i, row in visitor_type_data.iterrows():
        fig_type.add_annotation(x=row["Month"], y=row["Total Visitors"] + 500, text=str(row["Total Visitors"]), 
                                showarrow=False, font=dict(size=12, color="black"))
    
    st.plotly_chart(fig_type)

    # Trend Prediction
    st.header("Trend Prediction")

if uploaded_file is not None:
    # Dropdown to select visitor type
    visitor_types = df["Visitor Type"].unique().tolist()
    selected_visitor_type = st.selectbox("Select Visitor Type for Prediction", visitor_types)

    # Dropdown to select weather condition
    weather_options = ["All"] + df["Weather Condition"].unique().tolist()
    selected_weather = st.selectbox("Select Weather Condition for Prediction", weather_options)

    # Get predictions based on visitor type + weather condition
    future_trends = predict_trends(df, selected_visitor_type, selected_weather)

    # Prepare historical data for comparison
    historical_data = df[df["Visitor Type"] == selected_visitor_type]
    if selected_weather != "All":
        historical_data = historical_data[historical_data["Weather Condition"] == selected_weather]

    historical_data = historical_data.resample("M", on="Date")["Number of Visitors"].sum().reset_index()

    if future_trends.empty or historical_data.empty:
        st.warning("Not enough historical data for this combination. Try selecting 'All' for weather.")
    else:
        # Create dual-axis plot with increased figure size
        fig = go.Figure()

        # Add historical data (Actual Visitors - Solid Orange Line)
        fig.add_trace(go.Scatter(x=historical_data["Date"], 
                                 y=historical_data["Number of Visitors"],
                                 mode='lines+markers', 
                                 name="Actual Visitors", 
                                 line=dict(color="orange")))

        # Add future predictions (Predicted Visitors - Dotted Purple Line)
        fig.add_trace(go.Scatter(x=future_trends["Date"], 
                                 y=future_trends["Predicted_Visitors"],
                                 mode='lines+markers', 
                                 name="Predicted Visitors", 
                                 line=dict(color="purple", dash='dot')))

        # Adjust Y-axis dynamically to ensure graph is not cut off
        max_y_value = max(historical_data["Number of Visitors"].max(), 
                          future_trends["Predicted_Visitors"].max()) + 2000  # Buffer for better visibility

        # Customize layout with larger figure size and better spacing
        fig.update_layout(
            title=f"Past vs Future Visitor Trends for {selected_visitor_type} ({selected_weather} Weather)",
            xaxis_title="Date",
            yaxis_title="Number of Visitors",
            legend_title="Visitor Data",
            xaxis=dict(tickmode="array", tickformat="%b %Y"),  # Show Month-Year format
            yaxis=dict(range=[0, max_y_value]),  # Adjust Y-axis dynamically
            width=1000,  # Increased width
            height=600   # Increased height
        )

        st.plotly_chart(fig)