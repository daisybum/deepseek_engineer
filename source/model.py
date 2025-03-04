from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def load_pretrained_model(model_name: str):
    """주어진 Hugging Face 모델 이름에 해당하는 토크나이저와 모델을 불러옵니다."""
    # 토크나이저 및 모델 로드 (LLaMA-2, DeepSeek 등 지원)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset=None, output_dir="model_output"):
    """주어진 데이터셋으로 모델을 Trainer를 통해 미세조정하고, 결과를 output_dir에 저장합니다."""
    # 학습 하이퍼파라미터 설정
    training_args = TrainingArguments(
        output_dir=output_dir,          # 체크포인트 저장 경로&#8203;:contentReference[oaicite:6]{index=6}
        num_train_epochs=3,            # 에폭 수 (필요에 따라 조정)
        per_device_train_batch_size=8, # 배치 크기
        save_steps=500,                # 몇 스텝마다 모델 저장할지
        save_total_limit=2,            # 저장할 체크포인트 최대 개수
        evaluation_strategy="epoch" if eval_dataset is not None else "no",  # 평가 전략
        logging_steps=100,             # 로깅 빈도
        learning_rate=5e-5,            # 학습률 등 기타 하이퍼파라미터
        load_best_model_at_end=True    # 평가 기준 가장 좋은 모델 저장
    )
    # Trainer 초기화 (Trainer를 사용하면 분산 학습, 혼합정밀도 등도 자동 처리됩니다)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # 필요 시 데이터 collator 또는 compute_metrics 등을 전달 가능
    )
    # 미세 조정 실행
    trainer.train()  # 학습 시작&#8203;:contentReference[oaicite:7]{index=7}
    # 학습된 모델 저장 (토크나이저 포함)&#8203;:contentReference[oaicite:8]{index=8}
    trainer.save_model()  # output_dir 경로에 모델 가중치, 토크나이저 등을 저장
    return trainer

def load_trained_model(model_path: str):
    """save_model로 저장한 경로에서 모델과 토크나이저를 다시 불러옵니다."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

# 사용 예시
model_name = "meta-llama/Llama-2-7b-hf"  # 또는 "deepseek-ai/deepseek-vl-1.3b-base" 등
tokenizer, base_model = load_pretrained_model(model_name)         # 사전학습 모델 로드
# TODO: train_dataset와 eval_dataset를 준비 (Dataset 객체 또는 Tensor 등)
trainer = fine_tune_model(base_model, tokenizer, train_dataset, eval_dataset=None)  # 모델 미세조정
# 학습 완료 후 모델 저장됐으므로, 경로에서 다시 로드 가능
finetuned_model_path = trainer.args.output_dir
tokenizer, finetuned_model = load_trained_model(finetuned_model_path)  # 저장된 모델 불러오기
