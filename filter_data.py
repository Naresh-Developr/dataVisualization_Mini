# import pandas as pd

def filter_prompt(model, user_prompt, csv_overview):
    print("Filter Prompt")
    # Define the prompt with instructions and user input
    prompt = f"""
    You are an expert in pandas, and your job is to generate pandas filtering code.
    
    The user wants to filter a DataFrame based on the following prompt: '{user_prompt}'.
    The DataFrame has the following columns: {csv_overview}.
    
    Generate the pandas code needed to filter this DataFrame according to the user's request.

    Important:
    - Use parentheses around each condition in the filter to ensure compatibility.
    - Combine multiple conditions with & or | operators without extra brackets.
    - Make sure to use a single df[...] expression and avoid chaining conditions.

    Example format:

    df[(df['column1'] == 'value') & (df['column2'] > value)]

    Only output the code without any explanation.
    """
    
    # Use model's method for generating responses
    response = model.generate_content(prompt)
    print("DF Filter:", response.text)
    
    return response.text.strip()  # Returning only the code as a string


def clean_code(code):
    clean_code = []
    for line in code.split('\n'):
        # Example: Removing comments or markers that start with '#'
        if not line.strip().startswith("```"):  # Ignore comment lines
            clean_code.append(line)
    return "\n".join(clean_code)


def filter_data(model, user_prompt, csv_overview, df):
    pandas_code = filter_prompt(model, user_prompt, csv_overview)
    # cleaned_code = clean_code(pandas_code)
    
    # Execute the code to filter the DataFrame and store it in `filter_df`
    local_vars = {"df": df}
    exec(f"filter_df = {pandas_code}", {}, local_vars)
    
    # Retrieve `filter_df` from local_vars
    filter_df = local_vars["filter_df"]
    print(filter_df)
    return filter_df
