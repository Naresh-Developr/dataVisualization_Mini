import streamlit as st
import pandas as pd

# Set up the page layout
st.set_page_config(layout="wide")

# Title for the app
st.title("Interactive Data Visualization Tool")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Initialize layout with columns
col1, col2, col3 = st.columns([1, 2, 2])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display column names on the left (col1)
    col1.header("Columns in CSV")
    col1.write(df.columns.tolist())

    # Prompt section in the middle top (col2)
    col2.header("Enter Your Prompt")
    prompt = col2.text_input("Describe the visualization you'd like to see:")

    # Placeholder for visualized data directly below the prompt input
    col2.header("Visualized Data")
    visualized_data_placeholder = col2.empty()  # Placeholder for future visualizations

    # Display different types of charts on the right (col3)
    col3.header("Available Chart Types")
    
    # Placeholder chart names (for demonstration purposes)
    chart_types = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Plot", "Histogram"]

    # Display all chart types initially
    for chart in chart_types:
        col3.subheader(chart)
        col3.write("This is a placeholder for the " + chart.lower() + ".")
    
    # Future functionality: based on the prompt and data, we will update the visualized data here.
else:
    st.write("Please upload a CSV file to get started.")
