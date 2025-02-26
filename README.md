# **Draco National Park Visitor Dashboard** 

## **Project Overview**
The Draco National Park Visitor Dashboard is an interactive Streamlit-based data visualization tool designed to analyze visitor trends, visitor type distributions, and forecast future visitor trends based on historical data and weather conditions.

This project helps park analysts and decision-makers:

**Track visitor trends over time** (daily, weekly, monthly).
**Analyze visitor type breakdowns** (Camping, One-day Visit, RV Center).
**Predict future visitor trends** based on **weather and historical patterns**.

# **Features**
### **Visitor Analysis**
Line chart visualization to track daily, weekly, and monthly visitor trends.
Dynamic dropdown selection for different timeframes.
Implementation: Uses Plotly’s px.line() to create a time-series trend chart.
### **Visitor Type Analysis**
Stacked bar chart to show visitor type distribution by month.
Displays total visitors per type (Camping, One-day Visit, RV Center).
Implementation: Uses Plotly’s px.bar() with stacked mode for clear comparisons.
### **Trend Prediction**
Dual-axis line chart comparing historical data vs. predicted future trends.
Machine learning-based forecasting using Facebook Prophet.
Weather-based filtering to analyze the impact of Sunny, Rainy, or Snowy conditions on visitor trends.
Implementation: Uses Plotly’s go.Figure() to overlay actual and predicted visitor trends.

## **Installation & Setup**
### **Prerequisites**
Python 3.8+
Required libraries (streamlit, pandas, plotly, prophet)
install dependencies
run dashboard by command -  streamlit run app.py

## **How It Works**
### **Upload CSV File**
Users upload a CSV file containing visitor data.
The file is dynamically processed, ensuring real-time analysis.

## **Technologies Used**
Python (Data Processing & ML)
Streamlit (Dashboard UI)
Plotly (Interactive Visualizations)
Pandas (Data Handling)
Facebook Prophet (Time-Series Forecasting)


## **Conclusion**
This Visitor Dashboard provides an efficient and interactive way for Draco National Park analysts to track visitor trends, analyze visitor types, and predict future attendance using historical data and weather conditions. 
By leveraging machine learning and dynamic visualizations, this project enhances decision-making and park management.


