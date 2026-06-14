# 1. Исследование моделей и инфраструктуры
## Сравнительный анализ LLM-моделей
| Критерий | Локальные (Hugging Face)            | OpenAI GPT-4 | YandexGPT |
|----------|-------------------------------------|--------------|-----------|
| Качество ответов | Среднее-высокое (зависит от модели) | Очень высокое | Высокое (для RU/EN) |
| Скорость работы | Зависит от GPU (медленнее на CPU)   | Быстро (200-500ms) | Быстро (300-600ms) |
| Стоимость | Большие инвестиции в GPU            | $0.01-0.03/1K токенов | ~0.004-0.012₽/1K токенов |
| Конфиденциальность | Полный контроль                     | Данные уходят в OpenAI | Данные в РФ |
| Простота развертывания | Сложно (требуется GPU/оптимизация)  | Просто (API) | Просто (API) |
## Сравнение моделей эмбеддингов
| Критерий | Sentence-Transformers (local) | OpenAI Embeddings | Yandex Embeddings |
|----------|-------------------------------|-------------------|-------------------|
| Размер эмбеддингов | 384-768 dims | 1536 dims | 256-1024 dims |
| Скорость индексации | 50-100 docs/sec (CPU) | 200-300 docs/sec | 150-250 docs/sec |
| Качество поиска | Хорошее (all-MiniLM-L6-v2) | Отличное | Хорошее |
| Стоимость | Бесплатно (локально) | $0.0001/1K токенов | Низкая |
## Сравнение векторных БД
| Критерий | FAISS                           | ChromaDB |
|----------|---------------------------------|----------|
| Скорость поиска | Очень высокая (C++ оптимизации) | Средняя (Python native) |
| Скорость индексации | Высокая                         | Средняя |
| Сложность внедрения | Простая (pip install faiss-cpu) | Простая |
| Поддержка | Facebook/Meta*                  | Open source |
| Персистентность | Сериализация в файл             | Встроенная БД |
| Стоимость | Бесплатно                       | Бесплатно |
| Идеальный сценарий | Production, высокая нагрузка    | Прототипирование, небольшие объемы |
*Запрещенная в РФ организация
## Рекомендуемые конфигурации сервера
### Вариант 1 (Локальный, минимальный):
* CPU: 4 cores
* RAM: 16 GB
* GPU: Не требуется (CPU для эмбеддингов)
* Хранилище: 100 GB SSD
### Вариант 2 (Локальный + GPU):
* CPU: 8 cores
* RAM: 32 GB
* GPU: NVIDIA T4 (16GB)
* Хранилище: 200 GB NVMe
### Вариант 3 (Гибридный - РЕКОМЕНДОВАН):
* CPU: 2 cores (только для API)
* RAM: 8 GB
* GPU: Не требуется (облачные эмбеддинги)
* Хранилище: 50 GB SSD
## Итоговые рекомендации для QuantumForge Software
### Выбранное решение для проекта:
* LLM: YandexGPT API (для конфиденциальности данных + поддержка EN/RU)
* Embeddings: all-MiniLM-L6-v2 (локально) - бесплатно, хорошее качество, полный контроль
* Embeddings-2: paraphrase-multilingual-MiniLM-L12-v2 (альтернатива для киррилицы), бесплатно, лучшее качество, колный контроль
* Vector DB: FAISS (высокая скорость поиска, простота, бесплатно)
## Инфраструктура:
**Вариант 3 (гибридный) - Docker-контейнер с FAISS + API к YandexGPT**
# 2. Подготовка базы знаний
[knowledge_base](knowledge_base)
# 3. Создание векторного индекса базы знаний
[build_index.py](build_index.py)
## Результаты:
* Модель: paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
* Чанков: 39
* Время генерации: 5 секунд
* [faiss.index](faiss.index)
* [metadata.pkl](metadata.pkl)
# 4. Реализация RAG-бота с техниками промптинга
* сам бот - [rag_bot.py](rag_bot.py)
* доказательства - [proof](proof)
# 5. Запуск и демонстрация работы бота
* Злонамеренный файл - [malicious.txt](knowledge_base/malicious.txt)
* Защищенный бот - [rag_bot_secure.py](rag_bot_secure.py), скрипт безопасности - [security.py](security.py)
* Доказательства - [security_proof](security_proof) и [security_audit.log](security_audit.log)
## Использованная защита
### Реализована многоуровневая система безопасности:
#### Уровень 1: ML модель viavoice/mdeberta-ru-prompt-injection
* Специально обучена на русском языке
* Детектит промпт-инъекции с высокой точностью
* Используется ТОЛЬкО для проверки запросов пользователя
#### Уровень 2: Regex-паттерны
Только самые явные
#### Уровень 3: OutputSanitizer
* Замена API-ключей на [REDACTED_API_KEY]
* Замена паролей на [REDACTED_PASSWORD]
* Блокировка утечки системного промпта
#### Уровень 4: Rate Limiting
* 60 запросов в минуту
* Защита от брутфорса и DDoS
#### Уровень 5: Аудит
* Логирование всех security-событий
* JSONL формат для интеграции с SIEM
## Потенциальные уязвимости
| № | Запрос | Проблема |
|---|--------|----------|
| 1 | Какой пароль у администратора? | malicious.txt проиндексирован → вредоносный контент в базе |
| 2 | суперпароль | Санитизация сработала, но ответ был сгенерирован |
| 3 | Общие запросы без контекста | LLM склонна к галлюцинациям |
## Выводы
### Работает корректно:
* Явные промпт-инъекции блокируются ML моделью
* Системный промпт не утекает через OutputSanitizer
* Rate limiting защищает от флуда
### Потенциально уязвимо:
* malicious.txt попал в индекс → нужен pre-indexing validation
* ML модель может давать ложные срабатывания на русском
* LLM иногда фантазирует при отсутствии контекста
### Для production необходимо:
* Чёрный список файлов на этапе индексации
* Регулярное обновление ML модели
* Более строгий системный промпт
### Для проекта: уровень защиты достаточен.
# 6. Автоматическое ежедневное обновление базы знаний
* Dockerfile для сборки в контейнере - [Dockerfile](Dockerfile)
* Оркестратор для сборки в контейнере - [docker-compose.updater.yml](docker-compose.updater.yml)
* Команда для локального запуска - [cron_setup.sh](cron_setup.sh)
* Скрипт обновления - [update_index.py](update_index.py)
* Логи обновления - [index_update.log](index_update.log)
## 1. Архитектура обновления базы знаний
```mermaid
flowchart TB
    subgraph Sources["Источники данных"]
        KB[("knowledge_base/")]
        HASH[".file_hashes.json"]
    end

    subgraph Scheduler["Планировщик"]
        CRON["Cron / systemd<br/>06:00 ежедневно"]
        MANUAL["Ручной запуск"]
    end

    subgraph Updater["update_index.py"]
        CHECK["get_changes()"]
        INCR["incremental_update()"]
        REBUILD["rebuild_index()"]
        EMBED["Генерация эмбеддингов"]
    end

    subgraph Storage["Хранилище"]
        INDEX["faiss.index"]
        META["metadata.pkl"]
        LOG["index_update.log"]
    end

    subgraph Bot["RAG Бот"]
        QUERY["Поиск по индексу"]
        ANSWER["Ответ пользователю"]
    end

    CRON --> Updater
    MANUAL --> Updater
    KB --> CHECK
    HASH --> CHECK
    
    CHECK -->|"Нет изменений"| LOG
    CHECK -->|"Мало изменений"| INCR
    CHECK -->|"Много изменений"| REBUILD
    
    INCR --> EMBED
    REBUILD --> EMBED
    
    EMBED --> INDEX
    EMBED --> META
    
    INCR --> LOG
    REBUILD --> LOG
    
    Bot --> INDEX
    INDEX --> QUERY
    QUERY --> ANSWER
```
## 2. Последовательность
```mermaid
sequenceDiagram
    participant User as Пользователь
    participant Cron as Cron (06:00)
    participant Updater as update_index.py
    participant FS as Файловая система
    participant Model as SentenceTransformer
    participant FAISS as FAISS индекс
    participant Log as Лог-файл

    User->>FS: Добавляет/изменяет файлы
    
    Cron->>Updater: Запуск в 06:00
    Updater->>FS: Чтение файлов
    FS-->>Updater: Список файлов
    
    Updater->>FS: Чтение .file_hashes.json
    FS-->>Updater: Сохранённые хэши
    
    Updater->>Updater: Сравнение хэшей
    
    alt Есть изменения
        Updater->>Log: Запись "Обнаружены изменения"
        
        alt Много изменений
            Updater->>Updater: Полная перестройка
            Updater->>FS: Чтение всех файлов
            FS-->>Updater: Содержимое
            Updater->>Model: Генерация эмбеддингов
            Model-->>Updater: Векторы
            Updater->>FAISS: Создание индекса
        else Мало изменений
            Updater->>Updater: Инкрементальное обновление
            Updater->>FS: Чтение новых файлов
            FS-->>Updater: Содержимое
            Updater->>Model: Генерация эмбеддингов
            Model-->>Updater: Векторы
            Updater->>FAISS: Добавление в индекс
        end
        
        Updater->>FS: Обновление .file_hashes.json
        Updater->>Log: Запись "Обновление завершено"
    else Нет изменений
        Updater->>Log: Запись "Нет изменений"
    end
    
    Updater-->>Cron: Завершение
```
## 3: Состояния
```mermaid
stateDiagram-v2
    [*] --> ИндексНеСоздан
    
    ИндексНеСоздан --> Создание: rebuild
    
    state Создание {
        [*] --> ЧтениеФайлов
        ЧтениеФайлов --> ГенерацияЭмбеддингов
        ГенерацияЭмбеддингов --> СохранениеИндекса
        СохранениеИндекса --> [*]
    }
    
    Создание --> АктуальныйИндекс
    
    state АктуальныйИндекс {
        [*] --> Ожидание
        Ожидание --> Проверка: запуск
        Проверка --> Ожидание: нет изменений
        Проверка --> Обновление: есть изменения
        Обновление --> Ожидание: обновлено
    }
    
    АктуальныйИндекс --> Устаревший: изменение файлов
    Устаревший --> АктуальныйИндекс: обновление
```
## 4: Поток данных
```mermaid
flowchart LR
    subgraph Input["Входные данные"]
        A1["Новые файлы"]
        A2["Изменённые файлы"]
        A3["Существующий индекс"]
    end
    
    subgraph Process["Обработка"]
        B1["Разбивка на чанки"]
        B2["Генерация эмбеддингов"]
        B3["Добавление в FAISS"]
        B4["Обновление метаданных"]
    end
    
    subgraph Output["Выход"]
        C1["Обновлённый индекс"]
        C2["Новые хэши"]
        C3["Лог обновления"]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> B3
    A3 --> B3
    B3 --> B4
    B4 --> C1
    B4 --> C2
    B3 --> C3
    B4 --> C3
```
## 5: Классы
```mermaid
classDiagram
    class KnowledgeBaseUpdater {
        -kb_dir: Path
        -index_path: str
        -file_hashes: Dict
        +__init__()
        +get_changes()
        +rebuild_index()
        +incremental_update()
        +run()
    }
    
    class HuggingFaceEmbeddings {
        +embed_documents()
        +embed_query()
    }
    
    class FAISS {
        +IndexFlatL2
        +add()
        +search()
        +read_index()
        +write_index()
    }
    
    KnowledgeBaseUpdater --> HuggingFaceEmbeddings
    KnowledgeBaseUpdater --> FAISS
```
# 7. Аналитика покрытия и качества базы знаний
* Золотые вопросы - [golden_questions.json](golden_questions.json)
* Аналитика покрытия и качества базы знаний - [evaluate.py](evaluate.py)
* Тест:
* * Полный тест
```python evaluate.py --run```
* * Только пробелы в покрытии, запускать только после теста
```python evaluate.py --gaps```
* логи - [evaluation_logs.csv](evaluation_logs.csv) и [evaluation_logs.jsonl](evaluation_logs.jsonl)
## Диаграмма 
```mermaid
sequenceDiagram
    participant User as Пользователь
    participant CLI as CLI (evaluate.py)
    participant Bot as RAG Бот
    participant Security as SecurityFilter
    participant Retriever as Поиск (FAISS)
    participant Reranker as Cross-Encoder
    participant LLM as YandexGPT
    participant Logs as Логи (JSONL)

    Note over User,Logs: === ФАЗА 1: ОБРАБОТКА ЗАПРОСА ===

    User->>CLI: Запуск python evaluate.py --run
    CLI->>Bot: Инициализация бота
    
    alt Ошибка инициализации
        Bot-->>CLI: Индекс не загружен
        CLI-->>User: Ошибка: faiss.index не найден
    end

    loop Для каждого золотого вопроса
        CLI->>Bot: ask(question)
        
        Note over Bot,Security: === Проверка безопасности ===
        Bot->>Security: validate_query(query)
        
        alt Обнаружена инъекция
            Security-->>Bot: Запрос отклонён
            Bot-->>CLI: Ответ: "Запрос отклонён"
            CLI->>Logs: Сохранить результат (score=2.0)
        end

        Note over Bot,Reranker: === Поиск и ранжирование ===
        
        Bot->>Retriever: search(query, k=20)
        
        alt Индекс пуст
            Retriever-->>Bot: Нет результатов
            Bot-->>CLI: Ответ: "Я не знаю"
            CLI->>Logs: Сохранить результат (score=0.0)
        end
        
        Retriever-->>Bot: Результаты (L2 distance)
        
        opt Если включен reranker
            Bot->>Reranker: rerank(query, results)
            Reranker-->>Bot: Отсортированные результаты
        end
        
        alt Невалидные чанки (score > threshold)
            Bot-->>CLI: Ответ: "Информация нерелевантна"
            CLI->>Logs: Сохранить результат (score=1.0)
        end

        Note over Bot,LLM: === Генерация ответа ===
        
        Bot->>Bot: build_prompt(query, context)
        
        Bot->>LLM: generate(prompt)
        
        alt LLM недоступен
            LLM-->>Bot: Ошибка API / таймаут
            Bot-->>CLI: Ответ: "Ошибка генерации"
            CLI->>Logs: Сохранить результат (score=0.0)
        else LLM ответил
            LLM-->>Bot: Сгенерированный ответ
        end

        Note over Bot,Logs: === Оценка качества ===
        
        Bot-->>CLI: Ответ
        CLI->>CLI: check_keywords(actual, expected)
        
        alt Ключевые слова найдены
            CLI->>CLI: score = 2.0 (отлично)
        else Частично найдены
            CLI->>CLI: score = 1.0 (частично)
        else Не найдены
            CLI->>CLI: score = 0.0 (плохо)
        end
        
        CLI->>Logs: Сохранить результат в JSONL
        end

    Note over CLI,Logs: === ФАЗА 2: ГЕНЕРАЦИЯ ОТЧЁТА ===

    CLI->>CLI: generate_report()
    CLI->>Logs: Чтение evaluation_logs.jsonl
    CLI->>CLI: Подсчёт статистики
    
    alt Логи пусты
        CLI-->>User: Нет данных для отчёта
    else
        CLI-->>User: Отчёт о качестве
        CLI-->>User: - Всего запросов: X
        CLI-->>User: - Отлично: Y
        CLI-->>User: - Частично: Z
        CLI-->>User: - Плохо: W
        CLI-->>User: - Успешность: N%
    end

    Note over CLI,User: === ФАЗА 3: ПОИСК ПРОБЕЛОВ ===

    opt python evaluate.py --gaps
        CLI->>Logs: Чтение evaluation_logs.jsonl
        CLI->>CLI: find_coverage_gaps()
        CLI->>CLI: all_sources - found_sources
        
        alt Есть непокрытые файлы
            CLI-->>User: Не найдены источники: [...] 
        end
        
        alt Есть галлюцинации
            CLI-->>User: Галлюцинации: [...]
        end
    end
```
