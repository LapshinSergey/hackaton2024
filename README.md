# Сервис обработки текста и вопросов на базе FastAPI

Этот проект представляет собой приложение на FastAPI, которое предоставляет API для загрузки документов, извлечения и обработки текста, а также генерации ответов на вопросы пользователей с использованием продвинутых моделей обработки естественного языка.

## Оглавление

- [Особенности](#особенности)
- [Требования](#требования)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Запуск приложения](#запуск-приложения)
- [API Эндпоинты](#api-эндпоинты)
  - [`POST /upload`](#post-upload)
  - [`/ask` WebSocket](#ask-websocket)
  - [`POST /process_questions`](#post-process_questions)
  - [`/prompt` WebSocket](#prompt-websocket)
  - [`POST /vote`](#post-vote)
- [Используемые модели](#используемые-модели)
- [Замечания по безопасности](#замечания-по-безопасности)
- [Лицензия](#лицензия)

## Особенности

- **Загрузка файлов**: Принимает документы в форматах PDF, Word, PPTX и TXT.
- **Извлечение текста**: Извлекает и обрабатывает текст из загруженных документов.
- **Определение языка и перевод**: Определяет язык текста и при необходимости переводит его.
- **Сегментация текста**: Сегментирует текст с использованием эмбеддингов SBERT.
- **Извлечение ключевых слов**: Извлекает ключевые слова из текста для индексирования.
- **Интеграция с Elasticsearch**: Индексирует извлеченные данные в Elasticsearch для эффективного поиска.
- **Ответы на вопросы**: Предоставляет ответы на вопросы пользователей на основе загруженных документов и моделей общего знания.
- **Подсказки в реальном времени**: Предлагает подсказки ключевых слов через соединения WebSocket.
- **Обратная связь**: Принимает обратную связь от пользователей для улучшения системы со временем.

## Требования

- Python 3.7 или выше
- [Elasticsearch](https://www.elastic.co/downloads/elasticsearch)
- GPU с поддержкой CUDA (опционально, но рекомендуется для производительности)
- Следующие Python-пакеты:
  - `fastapi`
  - `uvicorn`
  - `nltk`
  - `PyPDF2`
  - `docx`
  - `pptx`
  - `python-magic`
  - `langdetect`
  - `elasticsearch`
  - `sentence-transformers`
  - `transformers`
  - `torch`
  - `scikit-learn`
  - `pandas`
  - `xlsxwriter`
  - `aiofiles`

## Установка

1. **Клонирование репозитория**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Создание виртуального окружения**

   ```bash
   python -m venv venv
   source venv/bin/activate  # На Windows используйте `venv\Scripts\activate`
   ```

3. **Установка зависимостей**

   ```bash
   pip install -r requirements.txt
   ```

4. **Загрузка данных NLTK**

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

5. **Настройка Elasticsearch**

   - Установите Elasticsearch с [официального сайта](https://www.elastic.co/downloads/elasticsearch).
   - Запустите службу Elasticsearch.
   - Убедитесь, что Elasticsearch доступен по адресу `https://localhost:9200`.

## Конфигурация

1. **Учетные данные Elasticsearch**

   Приложение подключается к Elasticsearch с использованием базовой аутентификации. Рекомендуется задавать учетные данные Elasticsearch через переменные окружения или файл конфигурации для безопасности.

   ```python
   import os

   es_url = os.getenv('ES_URL', 'https://localhost:9200')
   username = os.getenv('ES_USERNAME', 'your_username')
   password = os.getenv('ES_PASSWORD', 'your_password')
   ```

2. **Пути к моделям**

   Убедитесь, что пути к вашим предварительно обученным моделям указаны правильно. Приложение ожидает, что модели доступны локально или будут загружены, если их нет.

## Запуск приложения

Запустите приложение FastAPI с помощью Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Эндпоинты

### `POST /upload`

Загружает документ для обработки.

- **Допустимые типы файлов**: PDF, DOCX, PPTX, TXT
- **Запрос**: Мультимедийные данные формы с полем файла.
- **Ответ**: JSON, содержащий имя файла и контрольную сумму.

### `/ask` WebSocket

Эндпоинт для ответов на вопросы в реальном времени.

- **Протокол**: WebSocket
- **Использование**: Отправьте вопрос в виде текста, получите JSON-ответ с ответами и оценками релевантности.

### `POST /process_questions`

Обрабатывает пакет вопросов из загруженного Excel-файла.

- **Запрос**: Мультимедийные данные формы с Excel-файлом, содержащим столбец 'question'.
- **Ответ**: Excel-файл с заполненными ответами.

### `/prompt` WebSocket

Предоставляет подсказки ключевых слов на основе префикса.

- **Протокол**: WebSocket
- **Использование**: Отправьте строку-префикс, получите список предложенных ключевых слов.

### `POST /vote`

Записывает обратную связь от пользователей для улучшения системы.

- **Запрос**: JSON, содержащий поля `question`, `answer` и `score`.
- **Ответ**: Статусное сообщение в формате JSON.

## Используемые модели

- **SentenceTransformer**: `paraphrase-MiniLM-L6-v2` для эмбеддингов текста.
- **Языковые модели**:
  - **Vicuna**: Для продвинутого ответа на вопросы.
  - **FLAN-T5**: `google/flan-t5-base` для задач последовательность-последовательность.
- **Модели перевода**:
  - **MarianMT**: `Helsinki-NLP/opus-mt-ru-en` и `Helsinki-NLP/opus-mt-en-ru` для перевода между русским и английским.

## Замечания по безопасности

- **Учетные данные**: Код содержит жестко закодированные учетные данные для Elasticsearch, что небезопасно. Настоятельно рекомендуется перенести эти учетные данные в переменные окружения или безопасный файл конфигурации.
- **Проверка SSL**: Проверка сертификата SSL отключена в клиенте Elasticsearch (`verify_certs=False`). Включите проверку SSL в производственной среде.
- **Конфиденциальность данных**: Убедитесь, что все загруженные документы и данные пользователей обрабатываются в соответствии с применимыми законами и правилами о защите данных.

## Лицензия

Этот проект распространяется под лицензией [MIT](LICENSE).