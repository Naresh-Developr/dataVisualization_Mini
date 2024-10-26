def data_visualization_prompt(user_prompt, csv_data_overview):
    prompt = f"""
        You are an AI specialized in data visualization. Your task is to analyze the user's prompt and the structure of the provided CSV file to generate a list of the best chart suggestions. Return the response as structured JSON. Follow these specific guidelines:

        1. **Analyze CSV Structure**:
           - The CSV data contains columns as follows: {csv_data_overview}.
           - Review and analyze the column types (e.g., numeric, categorical, date) and relationships between columns to suggest appropriate visualizations.

        2. **Chart Type Suggestion**:
           - Based on the user prompt, recommend 5 chart types from the following options: 
             - Bar chart, Line chart, Pie chart, Scatter plot, Histogram.
           - Ensure that each recommendation aligns with the data type and user intent.

        3. **JSON Output Format**:
           - For each suggested chart type, provide the output in the following format as a list:

           [
               {{
                   "chart_type": "Bar chart",
                   "x_axis": ["<list of values for x-axis>"],
                   "y_axis": ["<list of values for y-axis>"]
               }},
               {{
                   "chart_type": "Line chart",
                   "x_axis": ["<list of values for x-axis>"],
                   "y_axis": ["<list of values for y-axis>"]
               }},
               {{
                   "chart_type": "Pie chart",
                   "x_axis": ["<list of values for x-axis>"],
                   "y_axis": ["<list of values for y-axis>"]
               }},
               {{
                   "chart_type": "Scatter plot",
                   "x_axis": ["<list of values for x-axis>"],
                   "y_axis": ["<list of values for y-axis>"]
               }},
               {{
                   "chart_type": "Histogram",
                   "x_axis": ["<list of values for x-axis>"],
                   "y_axis": ["<list of values for y-axis>"]
               }}
           ]

        4. **Do Not Provide Explanations**:
           - Output only the structured JSON response. Avoid any additional explanations or descriptions.

        5. **User Query**: 
           - The user's query: "{user_prompt}"
    """
    return prompt
