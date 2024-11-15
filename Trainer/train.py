import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer
from torch.utils.data import DataLoader, Subset
from LoadDataset import MusicGenDataset  # 기존 데이터셋 준비 코드 사용

# 데이터셋 폴더 경로 지정
dataset_folder = r"..\Data\Data_Processed"

# 데이터셋 및 DataLoader 생성 (적은 데이터만 사용)
dataset = MusicGenDataset(dataset_folder)
limited_dataset = Subset(dataset, range(5))  # 데이터 5개로 제한
dataloader = DataLoader(limited_dataset, batch_size=2, shuffle=True)

# T5 모델 로드
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 토크나이저 로드
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# 학습 가능한 프롬프트 벡터 추가
prompt_length = 10
prompt_embedding = torch.nn.Parameter(torch.randn(prompt_length, model.config.d_model))

# 모델의 입력 임베딩에 프롬프트 벡터 추가 (전체 토큰 임베딩 크기 조정 포함)
input_embeddings = model.get_input_embeddings()
new_weight = torch.cat((prompt_embedding, input_embeddings.weight), dim=0)  # 프롬프트 벡터를 첫 번째 차원에 추가
model.set_input_embeddings(torch.nn.Embedding.from_pretrained(new_weight, freeze=False))
model.resize_token_embeddings(len(new_weight))  # 토큰 임베딩 크기 재조정

# 옵티마이저 설정
optimizer = torch.optim.AdamW([prompt_embedding] + list(model.parameters()), lr=5e-5)

# 학습 설정 정의
training_args = TrainingArguments(
    output_dir=r'..\results\checkpoints',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=r'..\results\logs',
    save_total_limit=2,  # 모델 체크포인트 저장 개수 제한
    save_steps=500,  # 몇 스텝마다 모델 저장할지 설정
)

# Trainer 객체 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=limited_dataset,
    eval_dataset=limited_dataset,  # 여기서는 동일한 데이터셋을 평가에 사용, 실제 사용 시 검증 데이터셋 필요
    optimizers=(optimizer, None),
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['input_ids'] for f in data])  # 디코더 입력으로 동일한 input_ids 사용
    }
)

# 모델 학습 시작
if __name__ == "__main__":
    trainer.train()
    # 학습된 모델 저장
    model.save_pretrained(r'..\results\trained_model')
    print("Model saved to '..\\results\\trained_model'")
