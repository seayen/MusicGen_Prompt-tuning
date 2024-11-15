from transformers import T5ForConditionalGeneration, MusicgenForConditionalGeneration, T5Tokenizer
import torch

# 훈련시킨 T5 모델 로드
trained_t5_model_path = r'..\results\trained_model'
text_encoder = T5ForConditionalGeneration.from_pretrained(trained_t5_model_path)

# T5 모델 토크나이저 로드
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# MusicGen 모델 로드
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# 모델에 프롬프트 튜닝한 T5 텍스트 인코더 삽입
model.encoder = text_encoder.encoder

# 텍스트 프롬프트 설정
prompt = "a calm piano piece"
inputs = tokenizer(prompt, return_tensors="pt")

# 음악 생성
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    generated_audio = model.generate(**inputs, max_length=100)

# 생성된 오디오를 파일로 저장 (이미 WAV로 변환된 상태)
audio_data = generated_audio.squeeze().cpu().numpy()
import soundfile as sf
sf.write(r"..\results\testgen.wav", audio_data, 16000)
