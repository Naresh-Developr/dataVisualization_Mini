import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import chardet
import read_csv
import chart

# Configure page


# Define chart icons and their descriptions
CHART_ICONS = {
    'line': {'icon': '📈', 'description': 'Best for temporal data and trends'},
    'bar': {'icon': '📊', 'description': 'Good for comparing categories'},
    'scatter': {'icon': '📉', 'description': 'Shows relationships between variables'},
    'pie': {'icon': '🥧', 'description': 'Displays part-to-whole relationships'},
    'histogram': {'icon': '📶', 'description': 'Shows distribution of data'},
    'funnel': {'icon': '📦', 'description': 'Displays data distribution and outliers'}
}

def initialize_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    return model

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
        df, error = read_csv.read_csv_with_encoding(uploaded_file)

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
                    suggestions = chart.get_chart_suggestion(model, st.session_state.df, user_prompt)
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
                
                fig = chart.create_chart(
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
