import os
import argparse
import csv
import string

def extract_sentences(args):
    data = []
    sentences_file_path = os.path.join(args.save_txt_path, 'sentences.txt')

    with open(args.file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        index = 1

        print(str(index) + " rows found")

        for row in reader:
            if index != 0:
                text = row['sentence']
                cleaned_text = ' '.join(word.strip(string.punctuation) for word in text.split())
                data.append(cleaned_text)
            index += 1

    # TXT file for all cleaned sentences
    print("Creating TXT file")

    with open(sentences_file_path, 'w', encoding='utf-8') as f:
        for sentence in data:
            f.write(sentence + '\n')

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility script to extract cleaned sentences from Common Voice Datasets")
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_txt_path', type=str, default=None, required=True,
                        help='path to the dir where the TXT file is supposed to be saved')

    args = parser.parse_args()

    extract_sentences(args)
