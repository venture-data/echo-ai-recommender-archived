import argparse
from ranking import search_page_request, product_page_request, in_cart_request

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Call different functions from ranking.py")
    
    # Add subparsers for different functions
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for search_page_request function
    search_parser = subparsers.add_parser('search', help='Search for products based on a query')
    search_parser.add_argument('query', type=str, help='The search query to find products')

    # Subparser for product_page_request function
    product_parser = subparsers.add_parser('product', help='Get product recommendations based on product_id and user_id')
    product_parser.add_argument('product_id', type=str, help='The product ID')
    product_parser.add_argument('user_id', type=str, help='The user ID')

    # Subparser for in_cart_request function
    cart_parser = subparsers.add_parser('cart', help='Get in-cart recommendations based on product_id and user_id')
    cart_parser.add_argument('product_id', type=str, help='The product ID in the cart')
    cart_parser.add_argument('user_id', type=str, help='The user ID')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the appropriate function based on the command
    if args.command == 'search':
        # Call search_page_request with the query argument
        result = search_page_request(args.query)
        print("Search Results:")
        print(result)

    elif args.command == 'product':
        # Call product_page_request with the product_id and user_id arguments
        result = product_page_request(args.product_id, args.user_id)
        print("Product Page Recommendations:")
        print(result)

    elif args.command == 'cart':
        # Call in_cart_request with the product_id and user_id arguments
        result = in_cart_request(args.product_id, args.user_id)
        print("In-Cart Recommendations:")
        print(result)

if __name__ == '__main__':
    main()
