import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import chardet
from prompt import data_visualization_prompt

# Configure page
st.set_page_config(page_title="CSV Visualizer", layout="wide")

# Define chart icons and their descriptions
CHART_ICONS = {
    'line': {'icon': '📈', 'description': 'Best for temporal data and trends'},
    'bar': {'icon': '📊', 'description': 'Good for comparing categories'},
    'scatter': {'icon': '📉', 'description': 'Shows relationships between variables'},
    'pie': {'icon': '🥧', 'description': 'Displays part-to-whole relationships'},
    'histogram': {'icon': '📶', 'description': 'Shows distribution of data'},
    'box': {'icon': '📦', 'description': 'Displays data distribution and outliers'}
}

def initialize_gemini():
    """Initialize and return the Gemini AI model"""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Configure the model settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Initialize the model with configurations
        model = genai.GenerativeModel(
            model_name='gemini-pro',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return model
    
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {str(e)}")
        return None
class DataFilter:
    @staticmethod
    def get_filter_operators(dtype):
        """Return appropriate operators based on data type"""
        if pd.api.types.is_numeric_dtype(dtype):
            return ['>', '<', '=', '>=', '<=', 'between']
        elif pd.api.types.is_datetime64_dtype(dtype):
            return ['before', 'after', 'between', '=']
        else:
            return ['contains', 'equals', 'starts with', 'ends with', 'is null', 'is not null']

    @staticmethod
    def apply_filter(df, column, operator, value, value2=None):
        """Apply a single filter condition"""
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                value = pd.to_numeric(value) if value != '' else None
                if value2 is not None and value2 != '':
                    value2 = pd.to_numeric(value2)
            elif pd.api.types.is_datetime64_dtype(df[column]):
                value = pd.to_datetime(value) if value != '' else None
                if value2 is not None and value2 != '':
                    value2 = pd.to_datetime(value2)

            if operator == '>':
                return df[df[column] > value]
            elif operator == '<':
                return df[df[column] < value]
            elif operator == '=':
                return df[df[column] == value]
            elif operator == '>=':
                return df[df[column] >= value]
            elif operator == '<=':
                return df[df[column] <= value]
            elif operator == 'between':
                return df[(df[column] >= value) & (df[column] <= value2)]
            elif operator == 'contains':
                return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
            elif operator == 'equals':
                return df[df[column].astype(str).eq(str(value))]
            elif operator == 'starts with':
                return df[df[column].astype(str).str.startswith(str(value), na=False)]
            elif operator == 'ends with':
                return df[df[column].astype(str).str.endswith(str(value), na=False)]
            elif operator == 'is null':
                return df[df[column].isna()]
            elif operator == 'is not null':
                return df[df[column].notna()]
            elif operator == 'before':
                return df[df[column] < value]
            elif operator == 'after':
                return df[df[column] > value]
            else:
                return df
        except Exception as e:
            st.error(f"Error applying filter: {str(e)}")
            return df

def add_filter_ui(df):
    """Create and manage filter UI"""
    st.subheader("Data Filters")
    
    # Display current filters
    if st.session_state.filters:
        st.write("Active Filters:")
        for idx, filter_config in enumerate(st.session_state.filters):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                filter_text = f"{filter_config['column']} {filter_config['operator']} {filter_config['value']}"
                if filter_config.get('value2'):
                    filter_text += f" and {filter_config['value2']}"
                st.write(filter_text)
            with col2:
                if st.button("🗑️", key=f"delete_{idx}"):
                    st.session_state.filters.pop(idx)
                    st.experimental_rerun()
            with col3:
                if st.button("📝", key=f"edit_{idx}"):
                    st.session_state.editing_filter = idx
                    st.experimental_rerun()

    # Add new filter section
    with st.expander("Add New Filter", expanded=not bool(st.session_state.filters)):
        col1, col2 = st.columns(2)
        
        with col1:
            column = st.selectbox("Select Column", df.columns)
            operators = DataFilter.get_filter_operators(df[column].dtype)
            operator = st.selectbox("Select Operator", operators)
        
        with col2:
            if operator == 'between':
                value = st.text_input("Minimum Value")
                value2 = st.text_input("Maximum Value")
            elif operator in ['is null', 'is not null']:
                value = ''
                value2 = None
            else:
                value = st.text_input("Filter Value")
                value2 = None
        
        if st.button("Add Filter"):
            new_filter = {
                'column': column,
                'operator': operator,
                'value': value,
                'value2': value2
            }
            st.session_state.filters.append(new_filter)
            st.experimental_rerun()

def apply_filters(df):
    """Apply all active filters to the dataframe"""
    filtered_df = df.copy()
    if 'filters' in st.session_state and st.session_state.filters:
        for filter_config in st.session_state.filters:
            filtered_df = DataFilter.apply_filter(
                filtered_df,
                filter_config['column'],
                filter_config['operator'],
                filter_config['value'],
                filter_config.get('value2')
            )
    return filtered_df

def get_chart_suggestion(model, df, user_prompt):
    """Get chart suggestions from Gemini AI"""
    try:
        # Prepare the context for Gemini
        columns_info = {col: str(df[col].dtype) for col in df.columns}
        data_summary = {
            "total_rows": len(df),
            "columns": columns_info,
            "sample_data": df.head(5).to_dict()
        }
        
        # Create a structured prompt for Gemini
        structured_prompt = {
            "user_request": user_prompt,
            "data_context": data_summary,
            "task": "visualization_suggestion",
            "output_format": "json",
            "required_fields": ["chart_type", "x_axis", "y_axis"]
        }
        
        # Get response from Gemini
        response = model.generate_content(json.dumps(structured_prompt))
        
        if not response or not response.text:
            raise ValueError("No response received from Gemini")
        
        # Parse the response
        suggestions = json.loads(response.text)
        
        # Process suggestions
        processed_suggestions = []
        for suggestion in suggestions:
            chart_type = suggestion['chart_type'].lower()
            
            if chart_type not in CHART_ICONS:
                continue
            
            x_axis = suggestion['x_axis']
            y_axis = suggestion['y_axis']
            
            # Verify columns exist
            if x_axis not in df.columns or y_axis not in df.columns:
                continue
            
            processed_suggestions.append({
                'chart_type': chart_type,
                'x_axis': x_axis,
                'y_axis': y_axis
            })
        
        if not processed_suggestions:
            st.warning("No suitable visualization suggestions found. Please try a different query.")
            return None
            
        return processed_suggestions
        
    except Exception as e:
        st.error(f"Error in chart suggestion generation: {str(e)}")
        return None

def create_chart(chart_type, df, x_axis, y_axis):
    """Create a Plotly chart based on the specified type and parameters"""
    try:
        if chart_type == 'line':
            fig = px.line(df, x=x_axis, y=y_axis)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == 'pie':
            fig = px.pie(df, values=y_axis, names=x_axis)
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_axis)
        elif chart_type == 'box':
            fig = px.box(df, x=x_axis, y=y_axis)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'filters' not in st.session_state:
        st.session_state.filters = []
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'chart_suggestions' not in st.session_state:
        st.session_state.chart_suggestions = None
    if 'selected_chart' not in st.session_state:
        st.session_state.selected_chart = None

def main():
    """Main application function"""
    st.title("Advanced CSV Data Visualizer")
    initialize_session_state()
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🔍 Data Explorer", "⚙️ Filter Management"])
            
            # Tab 1: Visualization
            with tab1:
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.subheader("Available Columns")
                    for col in df.columns:
                        with st.expander(f"📊 {col}"):
                            st.write(f"Type: {df[col].dtype}")
                            st.write(f"Unique values: {len(df[col].unique())}")
                            if pd.api.types.is_numeric_dtype(df[col]):
                                st.write(f"Range: {df[col].min()} to {df[col].max()}")
                
                # Apply filters
                filtered_df = apply_filters(df)
                st.session_state.filtered_df = filtered_df
                
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
                            if model:
                                suggestions = get_chart_suggestion(model, filtered_df, user_prompt)
                                if suggestions:
                                    st.session_state.chart_suggestions = suggestions
                                    st.session_state.selected_chart = suggestions[0]
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                    
                    # Display visualization
                    if st.session_state.selected_chart:
                        fig = create_chart(
                            st.session_state.selected_chart['chart_type'],
                            filtered_df,
                            st.session_state.selected_chart['x_axis'],
                            st.session_state.selected_chart['y_axis']
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.subheader("Chart Selection")
                    for chart_type, info in CHART_ICONS.items():
                        if st.button(
                            f"{info['icon']} {chart_type.title()}", 
                            key=f"btn_{chart_type}",
                            help=info['description']
                        ):
                            if st.session_state.chart_suggestions:
                                matching_suggestion = next(
                                    (s for s in st.session_state.chart_suggestions if s['chart_type'] == chart_type),
                                    st.session_state.chart_suggestions[0]
                                )
                                if matching_suggestion:
                                    st.session_state.selected_chart = matching_suggestion
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == '__main__':
    main()