import streamlit as st
import pandas as pd
import google.generativeai as genai
import chardet

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

# Function to generate Pandas filtering code using AI
def generate_pandas_filter_code(model, user_prompt, df_columns):
    # Create the AI prompt for generating the Pandas filtering code
    ai_prompt = f"""
    You are an expert in pandas, and your job is to generate pandas filtering code.
    
    The user wants to filter a DataFrame based on the following prompt: '{user_prompt}'.
    The DataFrame has the following columns: {df_columns}.
    
    Generate the pandas code needed to filter this DataFrame according to the user's request.
    Only output the code without any explanation.
    """
    
    response = model.generate_content(ai_prompt)
    
    return response.text.strip()  # Return the generated Pandas code

# Function to execute generated Pandas code and filter the DataFrame
def filter_data_with_pandas_code(df, pandas_code):
    try:
        # Execute the generated Pandas code to filter the DataFrame
        filtered_df = eval(pandas_code, {'df': df, 'pd': pd})
        return filtered_df
    except Exception as e:
        st.error(f"Error executing the generated pandas code: {str(e)}")
        return None

# Main Streamlit function
def main():
    st.title("CSV Data Visualizer")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = ""

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df, error = read_csv_with_encoding(uploaded_file)

        if error:
            st.error(error)
            return

        st.session_state.df = df

        # Display columns
        st.write("## Available Columns")
        st.write(list(df.columns))

        # Input from the user
        st.write("## Describe the filter you want to apply")
        user_prompt = st.text_area(
            "Enter your filter prompt (e.g., 'Show me the high profits in the southern region where product's sale is greater than 1000')",
            height=100,
            placeholder="Write your filtering condition here"
        )

        if st.button("Generate Pandas Code and Filter Data"):
            try:
                # Initialize AI model
                model = initialize_gemini()

                # Generate Pandas code based on the user's prompt
                pandas_code = generate_pandas_filter_code(model, user_prompt, list(df.columns))
                st.write("### Generated Pandas Code")
                st.code(pandas_code, language='python')

                # Save the generated code to session state
                st.session_state.generated_code = pandas_code

                # Execute the Pandas code to filter the data
                filtered_df = filter_data_with_pandas_code(df, pandas_code)

                if filtered_df is not None:
                    st.session_state.filtered_df = filtered_df
                    st.write("### Filtered Data")
                    st.write(filtered_df)

            except Exception as e:
                st.error(f"Error generating pandas code: {str(e)}")

        # Option to display the filtered DataFrame
        if st.session_state.filtered_df is not None:
            st.write("## Filtered Data")
            st.write(st.session_state.filtered_df)

if __name__ == '__main__':
    main()
