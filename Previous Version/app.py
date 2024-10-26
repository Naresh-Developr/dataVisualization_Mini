import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import chardet

# Configure page
st.set_page_config(page_title="CSV Visualizer", layout="wide")

# Initialize Gemini API
def initialize_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    return model

def detect_encoding(file_content):
    """Detect the encoding of the file using chardet"""
    result = chardet.detect(file_content)
    return result['encoding']

def read_csv_with_encoding(uploaded_file):
    """Read CSV file with proper encoding detection"""
    try:
        # Read the uploaded file as bytes
        bytes_data = uploaded_file.getvalue()
        
        # Detect the encoding
        encoding = detect_encoding(bytes_data)
        
        # Try reading with detected encoding
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df, None
        except UnicodeDecodeError:
            # If detected encoding fails, try common encodings
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    return df, None
                except UnicodeDecodeError:
                    continue
            
            return None, "Unable to read the file with any common encoding. Please check the file encoding."
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

def get_chart_suggestion(model, columns, column_types):
    # Prompt engineering for chart suggestions
    prompt = f"""
    As a data visualization expert, suggest the best chart type (choose between 'line', 'bar', 'scatter') 
    based on these columns and their data types: {columns} with types {column_types}.
    Return your response as a JSON object with this structure:
    {{
        "chart_type": "line/bar/scatter",
        "x_axis": "SALES",
        "y_axis": "QUANTITYORDERED",
        "reasoning": "brief explanation of why this chart type is suitable"
    }}
    Only return the JSON object, no other text.
    """
    
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except:
        return {
            "chart_type": "bar",
            "x_axis": columns[0],
            "y_axis": columns[1] if len(columns) > 1 else columns[0],
            "reasoning": "Default visualization"
        }

def create_chart(chart_type, df, x_col, y_col):
    if chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    else:  # bar chart as default
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    return fig

def main():
    st.title("CSV Data Visualizer")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV with encoding detection
        df, error = read_csv_with_encoding(uploaded_file)
        
        if error:
            st.error(error)
            return
            
        st.session_state.df = df
        
        # Create layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Columns")
            # Display column information
            for col in st.session_state.df.columns:
                st.write(f"📊 {col}")
                
            # Column selection
            x_axis = st.selectbox("Select X-axis", st.session_state.df.columns)
            y_axis = st.selectbox("Select Y-axis", st.session_state.df.columns)
            
            if st.button("Get Chart Suggestion"):
                try:
                    model = initialize_gemini()
                    column_types = {col: str(st.session_state.df[col].dtype) 
                                  for col in [x_axis, y_axis]}
                    suggestion = get_chart_suggestion(model, [x_axis, y_axis], column_types)
                    
                    st.write("### Suggestion")
                    st.write(f"Chart type: {suggestion['chart_type']}")
                    st.write(f"Reasoning: {suggestion['reasoning']}")
                    
                    # Create and display the suggested chart
                    fig = create_chart(
                        suggestion['chart_type'],
                        st.session_state.df,
                        x_axis,
                        y_axis
                    )
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
        
        with col2:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)

if __name__ == "__main__":
    main()