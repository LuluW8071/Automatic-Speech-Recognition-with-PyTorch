### Run this script in the terminal 
Note: <br>
- The file path will differ and try to give absolute path to track JSONs filepath

    ```
    py create_commonvoice_jsons.py --file_path "audio\common_voice_dataset\en\validated.csv" --save_json_path "F:\SpeechRecognition\scripts" --audio "F:\SpeechRecognition\scripts\audio\common_voice_dataset\en\clips" --percent 10 --convert

    ```

- JSON format
```
[
    {
        "key": "F:/SpeechRecognition/scripts/clips/common_voice_en_37476269.wav",
        "text": "This led to both the Red Terror and the White Terror."
    }
]
```
