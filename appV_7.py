import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import chardet
from prompt import data_visualization_prompt

# Configure page
st.set_page_config(page_title="Kutty Tableau", layout="wide")

# Define chart icons and their descriptions
CHART_ICONS = {
    'line': {'icon': '📈', 'description': 'Best for temporal data and trends'},
    'bar': {'icon': '📊', 'description': 'Good for comparing categories'},
    'scatter': {'icon': '📉', 'description': 'Shows relationships between variables'},
    'pie': {'icon': '🥧', 'description': 'Displays part-to-whole relationships'},
    'histogram': {'icon': '📶', 'description': 'Shows distribution of data'},
    'funnel': {'icon': '📦', 'description': 'Displays data distribution and outliers'}
}

# Initialize Gemini API
def initialize_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    return model

def detect_encoding(file_content):
    result = chardet.detect(file_content)
    return result['encoding']

def read_csv_with_encoding(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()
        encoding = detect_encoding(bytes_data)
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df, None
        except UnicodeDecodeError:
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    return df, None
                except UnicodeDecodeError:
                    continue
            return None, "Unable to read the file with any common encoding."
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

# Filtering method integration
def generate_pandas_filter_code(model, user_prompt, df_columns):
    # Create the AI prompt for generating the Pandas filtering code
    ai_prompt = f"""
    You are an expert in pandas, and your job is to generate pandas filtering code.
    
    The user wants to filter a DataFrame based on the following prompt: '{user_prompt}'.
    The DataFrame has the following columns: {df_columns}.
    
    Generate the pandas code needed to filter this DataFrame according to the user's request.

    Important:
    - Use parentheses around each condition in the filter to ensure compatibility.
    - Combine multiple conditions with `&` or `|` operators without extra brackets.
    - Make sure to use a single `df[...]` expression and avoid chaining conditions.

    Example format:

    df[(df['column1'] == 'value') & (df['column2'] > value)]

    Only output the code without any explanation.
    """
    response = model.generate_content(ai_prompt)
    return response.text.strip()

def filter_data_with_pandas_code(df, pandas_code):
    try:
        filtered_df = eval(pandas_code, {'df': df, 'pd': pd})
        return filtered_df
    except Exception as e:
        st.error(f"Error executing the generated pandas code: {str(e)}")
        return None 

def main():
    st.title("CSV Data Visualizer")

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = ""

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df, error = read_csv_with_encoding(uploaded_file)
        if error:
            st.error(error)
            return
        st.session_state.df = df

        st.write("## Available Columns")
        st.write(list(df.columns))

        st.write("## Describe the filter you want to apply")
        user_prompt = st.text_area("Enter your filter prompt", height=100)

        if st.button("Generate Pandas Code and Filter Data"):
            model = initialize_gemini()
            pandas_code = generate_pandas_filter_code(model, user_prompt, list(df.columns))
            st.write("### Generated Pandas Code")
            st.code(pandas_code, language='python')

            st.session_state.generated_code = pandas_code
            filtered_df = filter_data_with_pandas_code(df, pandas_code)

            if filtered_df is not None:
                st.session_state.filtered_df = filtered_df
                st.write("### Filtered Data")
                st.write(filtered_df)

        if st.session_state.filtered_df is not None:
            st.write("## Filtered Data")
            st.write(st.session_state.filtered_df)

if __name__ == '__main__':
    main()
