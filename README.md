# Python AI Agent with Web Interface

🚀 Веб-интерфейс для AI-ассистента с возможностью выполнения Python-кода, управления файлами и работы с веб-контентом


## Особенности

- 🧠 Интеграция с GPT-4 через Helicone API
- 💻 Выполнение терминальных команд напрямую через интерфейс
- 🌐 Парсинг веб-страниц с помощью Playwright и BeautifulSoup
- 🔍 Поиск информации в интернете через Serper API
- 📁 Автоматическое управление файловой структурой проектов
- 📝 Подсветка синтаксиса Markdown и кода в реальном времени
- 🔄 WebSocket-подключение для мгновенного взаимодействия

## Быстрый старт

### Предварительные требования
- Python 3.10+
- Установленный Playwright: `playwright install`
- API ключи (добавить в .env файл):
  - `HELICONE_API_KEY`
  - `SERPER_API_KEY`

### Установка
```bash
git clone https://github.com/right-git/python-ai-agent.git
cd python-ai-agent
pip install -r requirements.txt
```

### Конфигурация

Создайте .env файл в корне проекта:
env:
```
HELICONE_API_KEY="ваш ключ"
OPENAI_API_KEY="ваш ключ"
SERPER_API_KEY="ваш ключ"
```

### Запуск
```bash
python main.py
```

Откройте http://localhost:8000 в браузере

# Ключевые возможности

## 🛠 Функциональные инструменты

- Выполнение shell-команд с обработкой stdin/stdout
- Автосохранение кода с валидацией структуры проекта
- Умный поиск в интернете с фильтрацией результатов
- Парсинг веб-страниц с антидетект-технологиями
- Автоматическая установка зависимостей
- Генерация документации (README.md, requirements.txt)

## Безопасность

- Изоляция всех операций в отдельной директории `./ai`
- Ограничение длины вывода команд (16k символов)
- Защита от бесконечных циклов
- Валидация URL перед парсингом

## Архитектура
```
├── main.py            # Основной FastAPI сервер
├── functions.py       # Реализация функциональных инструментов
├── config.py          # Конфигурация и настройки
├── index.html         # Веб-интерфейс чата
└── requirements.txt   # Зависимости
```

## Для разработчиков

### Расширение функционала

- Добавьте новую функцию в `functions.py`
- Зарегистрируйте инструмент в `TOOLS` (config.py)
- Обновите системный промпт при необходимости
- Протестируйте через WebSocket-подключение

### Используемые технологии

- **FastAPI** - высокопроизводительный веб-фреймворк. [Документация FastAPI](https://fastapi.tiangolo.com/)
- **Playwright** - автоматизация браузера с stealth-режимом. [Документация Playwright](https://playwright.dev/docs/intro)
- **Serper API** - быстрый поиск через Google. [Документация Serper API](https://serper.dev/docs)
- **Loguru** - продвинутое логирование. [Документация Loguru](https://loguru.readthedocs.io/en/stable/)
- **OpenAI** - API для работы с моделями искусственного интеллекта. [Документация OpenAI](https://platform.openai.com/docs/)
- **Helicone** - API для работы с OpenAI, оптимизированный для производительности. [Документация Helicone](https://docs.helicone.ai/)
- **BeautifulSoup4** - библиотека для парсинга HTML и XML документов. [Документация BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)


📌 Этот проект был создан в рамках обучающего видео на YouTube.  
🎥 Полная видеоинструкция: [https://www.youtube.com/watch?v=1BpziOgz0tk](https://www.youtube.com/watch?v=1BpziOgz0tk)

💡 Все вопросы и предложения приветствуются в комментариях к видео!
