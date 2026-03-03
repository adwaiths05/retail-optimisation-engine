import pandas as pd
import great_expectations as gx
import sys

def validate_instacart_data():
    context = gx.get_context()
    
    # Define expectations for products.csv
    df = pd.read_csv("./data/raw/products.csv")
    validator = context.sources.add_pandas_wrapper(
        name="instacart_datasource", data=df
    ).get_validator(expectation_suite_name="product_suite")

    # Schema Rules
    validator.expect_column_to_exist("product_id")
    validator.expect_column_values_to_be_of_type("product_id", "int64")
    validator.expect_column_values_to_be_between("price", min_value=0, max_value=100)
    
    results = validator.validate()
    if not results.success:
        print("❌ Data Validation Failed!")
        sys.exit(1)
    print("✅ Data Validation Passed.")

if __name__ == "__main__":
    validate_instacart_data()