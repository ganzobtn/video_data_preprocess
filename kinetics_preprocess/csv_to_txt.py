import csv

def csv_to_txt(csv_file_path, txt_file_path):
    with open(csv_file_path, 'r') as csv_file:
        with open(txt_file_path, 'w') as txt_file:
            # Create a CSV reader object
            csv_reader = csv.reader(csv_file)

            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Write each row to the text file
                print(row)
                txt_file.write(row[0]+ '\n')

# Example usage
csv_file_path = '/projects/videomaev2/datas/dgx/finetune/revised/wlasl_2000/test.csv'
txt_file_path = '/projects/data/wlasl_2000/annotations/test.txt'

csv_to_txt(csv_file_path, txt_file_path)