import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa = self.data[idx]
        prompt = qa["question"] + " " + qa["answer"]
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMa-base model on Q/A data.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the JSON file with Q/A data.')
    parser.add_argument('--model_name', type=str, default='lmsys/vicuna-7b-v1.5', help='Pretrained LLaMa model name or path.')
    parser.add_argument('--output_dir', type=str, default='./llama-base_fine-tuned', help='Directory to save the fine-tuned model.')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size per GPU.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X steps.')
    parser.add_argument('--quantization', action='store_true', help='Use 4-bit quantization.')

    return parser.parse_args()

def main():
    args = parse_args()

    # Проверка наличия GPU
    if not torch.cuda.is_available():
        raise ValueError("GPU не обнаружен. Пожалуйста, убедитесь, что GPU доступен и CUDA установлена.")

    # Загрузка токенизатора
    token = os.getenv('HF_AUTH_TOKEN')
    if not token:
        raise ValueError("Не найден токен Hugging Face. Установите переменную окружения 'HF_AUTH_TOKEN'.")
    
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name, use_auth_token=token)

    # Загрузка модели с квантизацией, если требуется
    if args.quantization:
        print("Используется 4-bit квантизация с bitsandbytes.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            device_map='auto'
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )

    # Включение градиентного контрольного шага и градиентов для входа
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Настройка LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Ранг матрицы адаптера
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Модули для применения LoRA (зависит от архитектуры модели)
    )

    model = get_peft_model(model, peft_config)

    # Подготовка данных
    dataset = QADataset(args.data_path, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Настройка аргументов тренировки
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,  # Увеличено для больших наборов данных
        save_steps=args.save_steps,
        fp16=True,
        save_total_limit=2,
        dataloader_num_workers=4,
    )

    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Обучение модели
    trainer.train()

    # Сохранение модели
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Модель успешно сохранена в {args.output_dir}")

if __name__ == "__main__":
    main()
