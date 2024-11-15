import os
import json
import torch
import torchaudio
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader

class MusicGenDataset(Dataset):
    def __init__(self, dataset_folder):
        # 폴더 내의 모든 JSON 및 MP3 파일 로드
        self.json_files = []
        self.audio_files = []
        for file_name in os.listdir(dataset_folder):
            if file_name.endswith('.json'):
                json_path = os.path.join(dataset_folder, file_name)
                mp3_path = json_path.replace('.json', '.mp3')
                if os.path.exists(mp3_path):
                    self.json_files.append(json_path)
                    self.audio_files.append(mp3_path)

        # T5 토크나이저 로드
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # JSON 파일 로드
        with open(self.json_files[idx], 'r') as f:
            data = json.load(f)

        # 텍스트 정보 추출
        keywords = data.get('keywords', '')
        description = data.get('description', '')
        moods = ', '.join(data.get('moods', []))
        prompt_text = f"{keywords}, {description}, {moods}"

        # 텍스트 토큰화
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        # 오디오 파일 로드
        waveform, sample_rate = torchaudio.load(self.audio_files[idx])

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'audio': waveform,
            'sample_rate': sample_rate
        }

if __name__ == "__main__":
    # 데이터셋 폴더 경로 지정
    dataset_folder = r"..\Data\Data_Processed"

    # 데이터셋 및 DataLoader 생성
    dataset = MusicGenDataset(dataset_folder)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 데이터셋 확인
    for batch in dataloader:
        print("Input IDs:", batch['input_ids'].shape)
        print("Attention Mask:", batch['attention_mask'].shape)
        print("Audio Shape:", batch['audio'].shape)
        print("Sample Rate:", batch['sample_rate'])
        break
