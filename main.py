import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import argparse
import sys

def load_data(file_path):
    """
    Load the dataset from the specified file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def perform_regression(data, dependent_var, independent_vars):
    """
    Perform linear regression using the specified dependent and independent variables.
    """
    X = data[independent_vars]
    y = data[dependent_var]
    
    # Add a constant to the independent variables matrix for intercept
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model

def main(file_path, dependent_var, independent_vars):
    data = load_data(file_path)
    
    # Check if the specified columns exist in the data
    for var in [dependent_var] + independent_vars:
        if var not in data.columns:
            print(f"Column '{var}' not found in the data.")
            sys.exit(1)
    
    model = perform_regression(data, dependent_var, independent_vars)
    
    print(model.summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform linear regression on a dataset.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing the data.')
    parser.add_argument('dependent_var', type=str, help='The dependent variable for the regression.')
    parser.add_argument('independent_vars', nargs='+', help='The independent variable(s) for the regression.')
    
    args = parser.parse_args()
    
    main(args.file_path, args.dependent_var, args.independent_vars)
