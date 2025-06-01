import argparse

def read_file(file_path):
    """Read the content of the file."""
    with open(file_path, 'r') as file:
        return file.readlines()

def process_data(lines):
    """Process the data and format it."""
    formatted_data = []
    for line in lines:
        # Example processing: strip whitespace and capitalize each line
        formatted_line = line.strip().capitalize()
        formatted_data.append(formatted_line)
    return formatted_data

def write_output(output_path, formatted_data):
    """Write the formatted data to an output file."""
    with open(output_path, 'w') as file:
        for line in formatted_data:
            file.write(line + '\n')

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process and format text data.')
    parser.add_argument('input_file', type=str, help='Path to the input text file')
    parser.add_argument('output_file', type=str, help='Path to the output text file')

    args = parser.parse_args()

    # Read, process, and write output
    try:
        lines = read_file(args.input_file)
        formatted_data = process_data(lines)
        write_output(args.output_file, formatted_data)
        print(f"Formatted data written to {args.output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()