import json
import os

def generate_json_files(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_title = None

    # Process all .mp3 files in the input directory
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".mp3"):
            # Extract base name without extension
            base_name = os.path.splitext(filename)[0]

            # Determine the song title based on the prefix (e.g., "2_" from "2_1")
            song_title_prefix = base_name.split('_')[0]

            # If the title prefix changes, prompt for new inputs
            if current_title != song_title_prefix:
                current_title = song_title_prefix
                artist = input(f"Enter the artist name for song {current_title}: ")
                description = input(f"Enter a description for the music {current_title}: ")
                keywords = input(f"Enter keywords for the music {current_title} (comma separated): ")
                title = input(f"Enter the title of the music {current_title}: ")
                instrument = input(f"Enter the instrument used for {current_title}: ")
                moods = input(f"Enter moods for {current_title} (comma separated): ").split(',')

                # Clean and format moods list
                moods = [mood.strip() for mood in moods if mood.strip()]

            # JSON structure
            json_data = {
                "key": "",
                "artist": artist,
                "sample_rate": 48000,
                "file_extension": "mp3",
                "description": description,
                "keywords": keywords,
                "duration": 15.0,
                "bpm": "",
                "genre": "holy",
                "title": title,
                "name": base_name,
                "instrument": instrument,
                "moods": moods
            }

            # Write the JSON file
            output_path = os.path.join(output_dir, f"{base_name}.json")
            with open(output_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
                print(f"JSON file created: {output_path}")


data_dir = '../Data/Data_processed'

generate_json_files(data_dir, data_dir)
