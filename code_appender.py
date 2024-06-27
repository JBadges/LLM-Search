import os

def find_and_format_py_files(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(f"File {file}\n\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")

if __name__ == "__main__":
    # Replace 'your_directory' with the path to the directory you want to search
    directory_to_search = './llm_search'
    # Replace 'output.txt' with the desired output file name
    output_filename = 'output.txt'
    find_and_format_py_files(directory_to_search, output_filename)
    print(f"Formatted contents have been written to {output_filename}")
