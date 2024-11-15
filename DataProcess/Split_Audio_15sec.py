from pydub import AudioSegment
import os


def split_all_mp3_in_directory(input_dir, output_dir, segment_duration=15):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .mp3 files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            file_path = os.path.join(input_dir, filename)
            split_mp3(file_path, output_dir, segment_duration)


def split_mp3(file_path, output_dir, segment_duration=15):
    # Load the audio file
    audio = AudioSegment.from_mp3(file_path)

    # Calculate the segment duration in milliseconds
    segment_duration_ms = segment_duration * 1000
    total_duration_ms = len(audio)

    # Extract the base name of the file (without extension)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Split the audio into segments
    for i in range(0, total_duration_ms, segment_duration_ms):
        segment = audio[i:i + segment_duration_ms]
        output_path = os.path.join(output_dir, f"{base_name}_{i // segment_duration_ms + 1}.mp3")
        segment.export(output_path, format="mp3")
        print(f"Segment saved to {output_path}")


input_data_dir = '../Data/Data_original'
output_data_dir = '../Data/Data_processed'

split_all_mp3_in_directory(input_data_dir, output_data_dir)
