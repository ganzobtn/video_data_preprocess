import csv
def read_txt_into_list(file_path):
    with open(file_path, 'r') as file:
        # Read lines from the file and remove leading/trailing whitespaces
        lines = [line.strip() for line in file.readlines()]
    return lines

def write_to_csv(file_path, lines):
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header with indexes
        csv_writer.writerow(['id', 'name'])
        
        # Write lines with indexes
        for index, line in enumerate(lines, start=1):
            csv_writer.writerow([index, line])


# Example usage:
file_path = '/projects/data/wlasl_2000/annotations/wlasl2000_classes.txt'  # Replace with your file path
lines_to_write = read_txt_into_list(file_path)

write_to_csv('/projects/data/wlasl_2000/annotations/wlasl2000_labels.csv', lines_to_write)