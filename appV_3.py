import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import chardet
from prompt import data_visualization_prompt  # Import the prompt generator

# Configure page
st.set_page_config(page_title="CSV Visualizer", layout="wide")

# Define chart icons and their disabled/enabled states
CHART_ICONS = {
    'line': '📈',
    'bar': '📊',
    'scatter': '📉',
    'pie': '🥧',
    'histo': '📶',
    'box': '📦'
}

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

def get_chart_suggestion(model, df, user_prompt):
    # Generate prompt using the external prompt function
    csv_overview = list(df.columns)
    prompt = data_visualization_prompt(user_prompt, csv_overview)  # Using prompt from prompt.py
    response = model.generate_content(prompt)
    
    print(response)
    try:
        return json.loads(response.text)
    except:
        return None  # Return None if the response is not valid JSON

def create_chart(chart_type, df, x_col, y_col, additional_params=None):
    if additional_params is None:
        additional_params = {}
    
    # Ensure that x_col and y_col are strings (not lists)
    if isinstance(x_col, list):
        x_col = x_col[0]  # Pick the first element if it's a list
    if isinstance(y_col, list):
        y_col = y_col[0]  # Pick the first element if it's a list
    
    color_col = additional_params.get('color')
    size_col = additional_params.get('size')
    
    if chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "pie":
        fig = px.pie(df, values=y_col, names=x_col, title=f"Distribution of {y_col}")
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_col, y=y_col, color=color_col, title=f"Distribution of {x_col}")
    elif chart_type == "box":
        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
    else:
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    st.title("CSV Data Visualizer")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'suggested_chart' not in st.session_state:
        st.session_state.suggested_chart = None
    if 'selected_chart_type' not in st.session_state:
        st.session_state.selected_chart_type = None

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df, error = read_csv_with_encoding(uploaded_file)

        if error:
            st.error(error)
            return

        st.session_state.df = df

        # Create layout with three columns
        col1, col2, col3 = st.columns([1, 2, 1])

        # Left column for displaying columns
        with col1:
            st.subheader("Columns")
            for col in st.session_state.df.columns:
                st.write(f"📊 {col}")

        # Middle column for user input
        with col2:
            st.subheader("Describe Your Visualization Need")
            user_prompt = st.text_area(
                "What would you like to visualize from this data?",
                height=100,
                placeholder="E.g., 'Show me the trend of sales over time' or 'Compare revenue across different categories'"
            )

            if st.button("Get Visualization Suggestion"):
                try:
                    model = initialize_gemini()
                    suggestion = get_chart_suggestion(model, st.session_state.df, user_prompt)
                    
                    if suggestion is None:
                        st.error("Error generating visualization: No valid suggestion received.")
                    else:
                        st.session_state.suggested_chart = suggestion
                        st.session_state.selected_chart_type = st.session_state.suggested_chart[0]['chart_type']
                        st.write("### AI Suggestion")
                        
                        # Check if suggested_chart is a list or a dictionary
                        if isinstance(st.session_state.suggested_chart, list):
                            for suggestion in st.session_state.suggested_chart:
                                st.write(f"Recommended Chart: {suggestion['chart_type'].title()} Chart")
                                st.write(f"X-axis: {suggestion['x_axis']}")
                                st.write(f"Y-axis: {suggestion['y_axis']}")
                        else:
                            st.write(f"Recommended Chart: {st.session_state.suggested_chart['chart_type'].title()} Chart")
                            st.write(f"X-axis: {st.session_state.suggested_chart['x_axis']}")
                            st.write(f"Y-axis: {st.session_state.suggested_chart['y_axis']}")
                            st.write("Reasoning:", st.session_state.suggested_chart['reasoning'])

                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")

            # Render the chart below the suggestion button
            if st.session_state.suggested_chart and st.session_state.selected_chart_type:
                # Find the chart type that is currently selected and render it
                for suggestion in st.session_state.suggested_chart:
                    if suggestion['chart_type'] == st.session_state.selected_chart_type:
                        fig = create_chart(
                            suggestion['chart_type'],
                            st.session_state.df,
                            suggestion['x_axis'],
                            suggestion['y_axis'],
                            suggestion.get('additional_params', {})
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # Right column for chart selection
        with col3:
            st.subheader("Chart Selection")

            # Extract the suggested chart types
            suggested_chart_types = []
            if isinstance(st.session_state.suggested_chart, list):
                suggested_chart_types = [s['chart_type'] for s in st.session_state.suggested_chart]
            elif st.session_state.suggested_chart:
                suggested_chart_types.append(st.session_state.suggested_chart['chart_type'])

            # Create chart selection grid
            cols = st.columns(3)
            for idx, (chart_type, icon) in enumerate(CHART_ICONS.items()):
                enabled = chart_type in suggested_chart_types  # Enable only AI-suggested chart types

                with cols[idx % 3]:
                    st.button(
                        f"{icon} {chart_type.title()}",
                        disabled=not enabled,  # Disable buttons for non-suggested chart types
                        key=f"chart_{chart_type}",
                        on_click=lambda chart_type=chart_type: st.session_state.update(selected_chart_type=chart_type)
                    )

if __name__ == "__main__":
    main()
