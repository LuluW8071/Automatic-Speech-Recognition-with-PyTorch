import os
import argparse
import csv

def main(args):
    with open(args.file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        total_files = sum(1 for _ in reader)  # Count the number of lines in the CSV
        csvfile.seek(0)

        for index, row in enumerate(reader, start=1):
            filename = row['path'].rpartition('.')[0] + ".wav"
            dst = os.path.join(args.save_json_path, 'clips', filename)

            # Print progress with percentage bar
            progress = (index / total_files) * 100
            print(f"Checking file {index}/{total_files} - Progress: {progress:.2f}%", end='\r')

            if not os.path.exists(dst):
                print(f"File not found: {filename}")
                continue

    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility script to check for converted and duration of audio files.")
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')

    args = parser.parse_args()

    main(args)



