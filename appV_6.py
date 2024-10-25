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
    'funnel': {'icon': '📦', 'description': 'Displays data distribution and outliers'}  # Updated key for funnel chart
}

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

def normalize_chart_type(chart_type):
    """Normalize chart type string to match CHART_ICONS keys"""
    chart_type = chart_type.lower().replace(' chart', '').replace('plot', '')
    if 'scatter' in chart_type:
        return 'scatter'
    return chart_type.strip()

def calculate_suitability_score(df, x_col, y_col, chart_type):
    """Calculate a suitability score for the suggested visualization"""
    score = 0
    
    # Check data types
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
    try:
        df[x_col] = pd.to_datetime(df[x_col], format="%Y-%m-%d")
        x_is_temporal = True
    except:
        x_is_temporal = False
    
    x_unique_ratio = len(df[x_col].unique()) / len(df[x_col])
    
    # Scoring rules
    if chart_type == 'line':
        score += 5 if x_is_temporal else 0
        score += 3 if y_is_numeric else 0
        score += 2 if x_unique_ratio > 0.1 else 0
    elif chart_type == 'bar':
        score += 4 if not x_is_temporal else 2
        score += 3 if y_is_numeric else 0
        score += 3 if x_unique_ratio < 0.2 else 0
    elif chart_type == 'scatter':
        score += 5 if x_is_numeric and y_is_numeric else 0
        score += 2 if x_unique_ratio > 0.5 else 0
    elif chart_type == 'pie':
        score += 4 if x_unique_ratio < 0.1 else 0
        score += 3 if y_is_numeric else 0
    elif chart_type == 'histogram':
        score += 5 if x_is_numeric else 0
    elif chart_type == 'box':
        score += 4 if y_is_numeric else 0
        score += 2 if not x_is_temporal else 0
    
    return score

def get_chart_suggestion(model, df, user_prompt):
    # Generate prompt using the external prompt function
    csv_overview = list(df.columns)
    prompt = data_visualization_prompt(user_prompt, csv_overview)
    response = model.generate_content(prompt)
    
    try:
        suggestions = json.loads(response.text)
        
        # Process and score each suggestion
        processed_suggestions = []
        for suggestion in suggestions:
            # Normalize chart type
            chart_type = normalize_chart_type(suggestion['chart_type'])
            
            # Skip if chart type is not supported
            if chart_type not in CHART_ICONS:
                continue
                
            # Ensure single column for axes
            x_axis = suggestion['x_axis'][0] if isinstance(suggestion['x_axis'], list) else suggestion['x_axis']
            y_axis = suggestion['y_axis'][0] if isinstance(suggestion['y_axis'], list) else suggestion['y_axis']
            
            # Calculate suitability score
            score = calculate_suitability_score(df, x_axis, y_axis, chart_type)
            
            processed_suggestions.append({
                'chart_type': chart_type,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'suitability_score': score,
                'reasoning': f"Suitability score: {score}/10"
            })
        
        # Sort by suitability score
        processed_suggestions.sort(key=lambda x: x['suitability_score'], reverse=True)
        return processed_suggestions
        
    except Exception as e:
        st.error(f"Error in AI response processing: {str(e)}")
        return None



def create_chart(chart_type, df, x_col, y_col, additional_params=None):
    if additional_params is None:
        additional_params = {}
    
    color_col = additional_params.get('color')
    size_col = additional_params.get('size')

    if isinstance(color_col, list):
        color_col = color_col[0]
    if isinstance(size_col, list):
        size_col = size_col[0]
    
    try:
        if chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                         title=f"{y_col} vs {x_col}")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                           title=f"{y_col} vs {x_col}")
        elif chart_type == "pie":
            fig = px.pie(df, values=y_col, names=x_col,
                        title=f"Distribution of {y_col}")
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col, y=y_col, color=color_col,
                             title=f"Distribution of {x_col}")
        elif chart_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} by {x_col}")
        elif chart_type == "funnel":  # Add funnel chart logic
            fig = px.funnel(df, x=x_col, y=y_col, title=f"Funnel Chart: {y_col} by {x_col}")
        else:  # Default to bar chart
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} vs {x_col}")
        
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def main():
    st.title("Kutty Tableau")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'chart_suggestions' not in st.session_state:
        st.session_state.chart_suggestions = None
    if 'selected_chart' not in st.session_state:
        st.session_state.selected_chart = None

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
            st.subheader("Available Columns")
            for col in st.session_state.df.columns:
                with st.expander(f"📊 {col}"):
                    st.write(f"Type: {df[col].dtype}")
                    st.write(f"Unique values: {len(df[col].unique())}")
                    if pd.api.types.is_numeric_dtype(df[col]):
                        st.write(f"Range: {df[col].min()} to {df[col].max()}")

        # Middle column for user input and visualization
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
                    suggestions = get_chart_suggestion(model, st.session_state.df, user_prompt)
                    if suggestions:
                        st.session_state.chart_suggestions = suggestions
                        st.session_state.selected_chart = suggestions[0]
                    else:
                        st.error("No suitable visualization suggestions found.")

                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")

            # Display chart suggestions and visualization
            if st.session_state.selected_chart:
                st.write(f"### Selected Chart: {st.session_state.selected_chart['chart_type'].title()} Chart")
                st.write(f"**Suitability Score**: {st.session_state.selected_chart['suitability_score']}/10")
                
                fig = create_chart(
                    st.session_state.selected_chart['chart_type'],
                    st.session_state.df,
                    st.session_state.selected_chart['x_axis'],
                    st.session_state.selected_chart['y_axis']
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # Right column for chart selection
        with col3:
            st.subheader("Chart Selection")
            
            # Create chart selection grid
            for chart_type, info in CHART_ICONS.items():
                is_recommended = False
                if st.session_state.chart_suggestions:
                    is_recommended = any(s['chart_type'] == chart_type for s in st.session_state.chart_suggestions[:3])
                
                button_style = "primary" if is_recommended else "secondary"
                if st.button(
                    f"{info['icon']} {chart_type.title()}", 
                    key=f"btn_{chart_type}",
                    type=button_style,
                    help=info['description']
                ):
                    matching_suggestion = next(
                        (s for s in st.session_state.chart_suggestions if s['chart_type'] == chart_type),
                        st.session_state.chart_suggestions[0] if st.session_state.chart_suggestions else None
                    )
                    if matching_suggestion:
                        st.session_state.selected_chart = matching_suggestion

if __name__ == '__main__':
    main()
