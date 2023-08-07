import os
import argparse
import json
import random
import csv
import subprocess
from pydub import AudioSegment

def main(args):
    data = []
    percent = args.percent
    clips_directory = os.path.join(args.save_json_path, 'clips')

    if not os.path.exists(clips_directory):
        os.makedirs(clips_directory)
    
    with open(args.file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        length = sum(1 for _ in reader)  # Count the number of lines in the CSV
        csvfile.seek(0)
        index = 1
        # print(reader)

        if args.convert:
            print(str(length) + " files found")
            # print("ok1")
            # print(type(reader))
        
        #read csv file and convert files of mp3 to .wav 
        i=0 
        for row in reader:
            if i!=0:
                # print(row.keys())
                file_name = row['path']
                filename = file_name.rpartition('.')[0] + ".wav"
                text = row['sentence']
                # print(file_name)
                # print(file_name[1])

                if args.convert:
                    data.append({
                        "key": os.path.join(args.save_json_path, 'clips', filename).replace('\\', '/'),
                        "text": text
                    })


                    print(f"Converting file {index}/{length} to wav ------------ ({(index/length)*100:.3f}%)", end="\r")


                    src = os.path.join(args.audio,file_name)
                    dst = os.path.join(args.save_json_path, 'clips', filename)
                    # print(src)

                    """
                    Note:
                    The conversion process using nvenc codec and cpu is same for audio files 
                    """
                    # Use FFmpeg with Nvidia codec for audio conversion
                    # os.system(f'ffmpeg -i "{src}" -c:a aac -strict experimental -b:a 192k -c:v copy "{dst}"')

                    # Load the MP3 audio
                    sound = AudioSegment.from_mp3(src)
                    duration_in_seconds = len(sound) / 1000
                    
                    if duration_in_seconds > 30:
                        print(f"Skipping file {index}/{length}: Duration > 30 seconds")
                    else:
                        print(f"Converting file {index}/{length} to wav", end="\r")
                        sound = sound.set_frame_rate(32000)  # Set frame rate (bitrate) to 32 kbps
                        
                        # Idk how above line works but somehow the wav audio bitrate is set to 512Kbps

                        sound.export(dst, format="wav")  # Export the modified sound to wav
                        index = index + 1
                else:
                    data.append({
                        "key": os.path.join(args.save_json_path, 'clips', filename).replace('\\', '/'),
                        "text": text
                    })
            i+=1

    random.shuffle(data)

    #JSON for train and test
    print("creating JSON's")

    train_data = data[:int(length * (1 - percent / 100))]
    test_data = data[int(length * (1 - percent / 100)):]

    with open(os.path.join(args.save_json_path, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(os.path.join(args.save_json_path, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility script to convert Common Voice into WAV and create training and test JSON files for speech recognition.")
    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    # New parser added
    parser.add_argument('--audio', type=str, default=None, required=True,
                        help='audio clips path')

    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')

    args = parser.parse_args()

    main(args)