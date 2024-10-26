from prompt import data_visualization_prompt
import pandas as pd
import json
from appV_6 import CHART_ICONS
import plotly.express as px
import filter_data

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
    filter_df = filter_data.filter_data(model, user_prompt, csv_overview, df)
    prompt = data_visualization_prompt(user_prompt, list(filter_df.columns))
    response = model.generate_content(prompt)
    
    try:
        # Load JSON response
        suggestions = json.loads(response.text)
        processed_suggestions = []

        # Process each suggestion
        for suggestion in suggestions:
            # Check if suggestion contains necessary fields
            if 'chart_type' in suggestion and 'x_axis' in suggestion and 'y_axis' in suggestion:
                chart_type = normalize_chart_type(suggestion['chart_type'])
                
                # Skip unsupported chart types
                if chart_type not in CHART_ICONS:
                    continue
                
                # Extract x_axis and y_axis, ensuring they're not lists of values
                x_axis = suggestion['x_axis'][0] if isinstance(suggestion['x_axis'], list) else suggestion['x_axis']
                y_axis = suggestion['y_axis'][0] if isinstance(suggestion['y_axis'], list) else suggestion['y_axis']
                
                # Handle unexpected y_axis fields
                if y_axis == 'Frequency': 
                    continue  # Skip unsupported or erroneous fields
                
                # Calculate suitability score
                score = calculate_suitability_score(df, x_axis, y_axis, chart_type)
                
                # Add to processed suggestions
                processed_suggestions.append({
                    'chart_type': chart_type,
                    'x_axis': x_axis,
                    'y_axis': y_axis,
                    'suitability_score': score,
                    'reasoning': f"Suitability score: {score}/10"
                })
        
        # Sort by suitability score
        processed_suggestions.sort(key=lambda x: x['suitability_score'], reverse=True)
        print('Processed Suggestions:', processed_suggestions)
        return processed_suggestions, filter_df
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Log detailed error message
        print(f"Error in AI response processing: {str(e)}")
        return None
    except Exception as e:
        print(f"Error in AI response processing: {str(e)}")
        return None



def create_chart(chart_type, df_filter, x_col, y_col, additional_params=None):
    print('\nCreating chart')
    print("Filterd Datafram:\n",df_filter)
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
            fig = px.line(df_filter, x=x_col, y=y_col, color=color_col, 
                         title=f"{y_col} vs {x_col}")
        elif chart_type == "scatter":
            fig = px.scatter(df_filter, x=x_col, y=y_col, color=color_col, size=size_col,
                           title=f"{y_col} vs {x_col}")
        elif chart_type == "pie":
            fig = px.pie(df_filter, values=y_col, names=x_col,
                        title=f"Distribution of {y_col}")
        elif chart_type == "histogram":
            fig = px.histogram(df_filter, x=x_col, y=y_col, color=color_col,
                             title=f"Distribution of {x_col}")
        elif chart_type == "box":
            fig = px.box(df_filter, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} by {x_col}")
        elif chart_type == "funnel":  # Add funnel chart logic
            fig = px.funnel(df_filter, x=x_col, y=y_col, title=f"Funnel Chart: {y_col} by {x_col}")
        else:  # Default to bar chart
            fig = px.bar(df_filter, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} vs {x_col}")
        
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        return fig
    except Exception as e:
        # st.error(f"Error creating chart: {str(e)}")
        return None