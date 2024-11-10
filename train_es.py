import argparse
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from elasticsearch import Elasticsearch
import warnings

warnings.filterwarnings("ignore")  # Игнорируем предупреждения SSL и другие

class QADataset(Dataset):
    def __init__(self, es_client, index_name, tokenizer, max_length=512):
        self.es_client = es_client
        self.index_name = index_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data_from_es()

    def load_data_from_es(self):
        es_query = {"query": {"match_all": {}}}
        res = self.es_client.search(
            index=self.index_name,
            body=es_query,
            size=10000  # Увеличьте при необходимости
        )
        data = [hit['_source'] for hit in res['hits']['hits']]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa = self.data[idx]
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        score = qa.get('score', '')
        prompt = f"Question: {question}\nAnswer (Score: {score}):"
        target = f" {answer}"
        # Токенизация подсказки и целевого ответа отдельно
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        target_tokens = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        # Объединение токенов
        input_ids = torch.cat([prompt_tokens['input_ids'], target_tokens['input_ids']], dim=1).squeeze()
        attention_mask = torch.cat([prompt_tokens['attention_mask'], target_tokens['attention_mask']], dim=1).squeeze()
        # Создание меток, игнорируя часть подсказки
        labels = input_ids.clone()
        labels[:prompt_tokens['input_ids'].size(1)] = -100  # Игнорируем токены подсказки

        # Обрезка или дополнение до max_length
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        elif input_ids.size(0) < self.max_length:
            padding_length = self.max_length - input_ids.size(0)
            input_ids = torch.cat([input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model on Q/A data with scores.")
    parser.add_argument('--model_name', type=str, default='lmsys/vicuna-3b-v1.5', help='Pretrained model name or path.')
    parser.add_argument('--output_dir', type=str, default='./fine-tuned-model', help='Directory to save the fine-tuned model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size per GPU.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length.')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X steps.')
    parser.add_argument('--quantization', action='store_true', help='Use 8-bit quantization.')
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=token)

    # Загрузка модели с квантизацией, если требуется
    if args.quantization:
        print("Используется 8-битная квантизация с bitsandbytes и offloading на CPU.")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            device_map={'': 'cpu'},  # Загрузка модели на CPU
            use_auth_token=token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            use_auth_token=token
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
        target_modules=["q_proj", "v_proj"]  # Модули для применения LoRA
    )

    model = get_peft_model(model, peft_config)

    # Подключение к Elasticsearch
    es_url = 'https://localhost:9200'
    username = 'elastic'
    password = 'Aetoa1Tahyor'

    es_client = Elasticsearch(
        es_url,
        http_auth=(username, password),
        verify_certs=False,
        ssl_show_warn=False
    )

    # Подготовка данных
    dataset = QADataset(es_client, 'learning-score', tokenizer, max_length=args.max_length)

    # Настройка аргументов тренировки
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Настройте при необходимости
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=args.save_steps,
        fp16=True,
        save_total_limit=2,
        dataloader_num_workers=4,
        optim="paged_adamw_8bit" if args.quantization else "adamw",
        report_to="none",  # Отключаем отчеты в WandB и других сервисах
    )

    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Обучение модели
    trainer.train()

    # Сохранение модели
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Модель успешно сохранена в {args.output_dir}")

if __name__ == "__main__":
    main()
