import argparse
import pandas as pd
from ranking import search_page_request

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the product search application.")
    parser.add_argument('--searching', type=str, help='The search query for finding products', required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the search function with the provided query
    query = args.searching
    result_df = search_page_request(query)

    # Display the results
    print(result_df)

if __name__ == '__main__':
    main()
