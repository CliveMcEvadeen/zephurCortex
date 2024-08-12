# main.py

import argparse
from src.refactoring.refactoring_engine import RefactoringEngine
import logging

def initialize():
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Initializing ZephyrCortex...")

def main():
    parser = argparse.ArgumentParser(description='ZephyrCortex - Autonomous Cognitive System')
    parser.add_argument('--code', type=str, required=True, help='Code snippet to refactor')

    args = parser.parse_args()

    # Initialize refactoring engine
    refactoring_engine = RefactoringEngine()

    try:
        # Refactor the provided code snippet
        refactored_code = refactoring_engine.refactor_code(args.code)
        logging.info("Refactored Code:\n%s", refactored_code)
        print("Refactored Code:\n", refactored_code)
    except ValueError as ve:
        logging.error("ValueError: %s", str(ve))
        print(f"Error: {ve}")
    except FileNotFoundError as fnf:
        logging.error("FileNotFoundError: %s", str(fnf))
        print(f"Error: File not found - {fnf.filename}")
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    initialize()
    main()
