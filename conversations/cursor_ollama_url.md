# Настройка бота для подключения к ollama
_Exported on 03.12.2025 at 19:22:58 GMT+3 from Cursor (1.7.54)_

---

**User**

нужно чтоб бот связывался с ollama всегда по base_url="http://ollama:11434"
а не по локалхост (в vector_store.py я уже внес правки)

---

**Cursor**

Я сначала посмотрю `vector_store.py`, `bot.py` и `docker-compose.yml`, чтобы найти все места, где ollama вызывается через localhost, затем изменю их на `http://ollama:11434` и проверю на ошибки. После правок кратко опишу, что именно поменял.

Сейчас настрою использование `http://ollama:11434` в основном коде бота через клиент Ollama и проверю файл на ошибки.

```python
import os
import asyncio
import json
import uuid
from typing import Any, Dict, List

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from dotenv import load_dotenv

from memory import Memory_Manager
from vector_store import VectorStore
from mcp_tools.mcp_weather import WeatherInput, WeatherOutput, get_weather
from mcp_tools.search_duckduckgo import (
    DuckDuckGoInput,
    DuckDuckGoOutput,
    duckduckgo_search,
)
from mcp_tools.mcp_notion import (
    NotionSearchInput,
    NotionSearchOutput,
    NotionGetPageInput,
    NotionPageContent,
    NotionCreatePageInput,
    NotionCreatePageOutput,
    notion_search,
    notion_get_page,
    notion_create_page,
)

import ollama


load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Явно указываем base_url для доступа к сервису Ollama внутри docker-compose
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
memory = Memory_Manager("/app/data/chat_memory.db")
vector_db = VectorStore()


class AddToVectorStates(StatesGroup):
    waiting_for_text: State = State()


SYSTEM_PROMPT = """Ты - полезный AI-ассистент в Telegram боте. Твои характеристики:

1. Отвечай вежливо и дружелюбно на русском языке.
2. Будь кратким, но информативным.
3. Если не знаешь ответа — честно говори об этом.
4. Не придумывай информацию.
5. Форматируй ответы для удобства чтения в Telegram.

ВАЖНО: Сначала отвечай на вопрос самостоятельно, используя свои знания. 
Используй инструменты ТОЛЬКО когда это действительно необходимо:
- Для получения ТЕКУЩЕЙ погоды (не исторической или общей информации о климате)
- Для поиска АКТУАЛЬНЫХ новостей или свежей информации из интернета
- Для работы с Notion (поиск, чтение, создание страниц)

НЕ используй инструменты для:
- Простых вопросов и приветствий
- Общих знаний, которые ты уже знаешь
- Философских или абстрактных вопросов
- Разговоров и беседы

Тебе доступны инструменты:

1) get_weather(city: string)
   — ТОЛЬКО для получения ТЕКУЩЕЙ погоды в конкретном городе.

2) duckduckgo_search(query: string)
   — ТОЛЬКО для поиска АКТУАЛЬНОЙ информации, новостей или свежих данных из интернета.

3) notion_search(query: string)
   — Для поиска страниц в Notion по запросу.

4) notion_get_page(page_id: string)
   — Для получения содержимого страницы Notion по ID.

5) notion_create_page(parent_id: string, title: string, content?: string)
   — Для создания новой страницы в Notion.

Отвечай в JSON формата OpenAI function calling.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Используй ТОЛЬКО для получения ТЕКУЩЕЙ погоды в конкретном городе. НЕ используй для общих вопросов о климате или исторических данных.",
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города на любом языке",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Используй ТОЛЬКО для поиска АКТУАЛЬНЫХ новостей, свежей информации или данных, которые могут измениться. НЕ используй для общих знаний, простых вопросов или разговоров.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос для актуальной информации из интернета",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notion_search",
            "description": "Ищет страницы в Notion по текстовому запросу",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос для поиска страниц в Notion",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notion_get_page",
            "description": "Получает содержимое страницы Notion по её ID",
            "parameters": {
                "type": "object",
                "required": ["page_id"],
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "ID страницы Notion для получения содержимого",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notion_create_page",
            "description": "Создаёт новую страницу в Notion. Если parent_id не указан, используется PARENT_ID из .env",
            "parameters": {
                "type": "object",
                "required": ["title"],
                "properties": {
                    "parent_id": {
                        "type": "string",
                        "description": "ID родительской страницы или базы данных (опционально, если не указан используется PARENT_ID из .env)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Заголовок новой страницы",
                    },
                    "content": {
                        "type": "string",
                        "description": "Содержимое страницы (текст, опционально)",
                    },
                },
            },
        },
    },
]


async def call_ollama(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Запускает обращение к Ollama в отдельном потоке, чтобы не блокировать event-loop."""
    return await asyncio.to_thread(
        lambda: ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
    )


def _parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        return json.loads(raw_args)
    raise ValueError("Неверный формат аргументов инструмента")


async def _execute_tool_call(
    tool_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    if tool_name == "get_weather":
        weather_input = WeatherInput(**arguments)
        result: WeatherOutput = await asyncio.to_thread(get_weather, weather_input)
        return result.model_dump()

    if tool_name == "duckduckgo_search":
        search_input = DuckDuckGoInput(**arguments)
        result: DuckDuckGoOutput = await asyncio.to_thread(
            duckduckgo_search, search_input
        )
        return result.model_dump()

    if tool_name == "notion_search":
        search_input = NotionSearchInput(**arguments)
        result: NotionSearchOutput = await asyncio.to_thread(
            notion_search, search_input
        )
        return result.model_dump()

    if tool_name == "notion_get_page":
        page_input = NotionGetPageInput(**arguments)
        result: NotionPageContent = await asyncio.to_thread(notion_get_page, page_input)
        return result.model_dump()

    if tool_name == "notion_create_page":
        create_input = NotionCreatePageInput(**arguments)
        result: NotionCreatePageOutput = await asyncio.to_thread(
            notion_create_page, create_input
        )
        return result.model_dump()

    raise ValueError(f"Инструмент {tool_name} не поддерживается")


async def llm_ollama(messages: List[Dict[str, Any]]) -> str:
    """Контролирует цикл LLM + инструменты до получения финального ответа."""
    conversation: List[Dict[str, Any]] = list(messages)

    for _ in range(5):  # предохранитель от бесконечных запросов инструментов
        response = await call_ollama(conversation)
        assistant_message = response.get("message", {})
        conversation.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            return assistant_message.get("content", "").strip()

        for call in tool_calls:
            function_meta = call.get("function", {})
            tool_name = function_meta.get("name")
            arguments = _parse_tool_arguments(function_meta.get("arguments"))

            try:
                tool_result = await _execute_tool_call(tool_name, arguments)
                tool_payload = json.dumps(tool_result, ensure_ascii=False)
            except Exception as exc:
                tool_payload = json.dumps({"error": str(exc)}, ensure_ascii=False)

            conversation.append(
                {"role": "tool", "tool_name": tool_name, "content": tool_payload}
            )

    return "Не удалось получить ответ от модели. Попробуй повторить запрос позже."


@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer(f"Привет, {message.from_user.first_name}! Я бот-помощник.")


@dp.message(Command("help"))
async def help_command(message: Message):
    await message.answer(
        "Я — персональный ассистент с памятью и доступом к инструментам:\n"
        "• DuckDuckGo — нахожу свежие данные в интернете.\n"
        "• Notion — читаю или создаю страницы в твоём рабочем пространстве.\n"
        "• Погода — сообщаю актуальные условия в любом городе.\n\n"
        "Просто задай вопрос текстом, например:\n"
        "— «Какая погода в Берлине?»\n"
        "— «Найди последнюю новость про автономные авто»\n"
        "— «Создай заметку в Notion: идеи для контента»\n\n"
        "Команды:\n"
        "/new_chat — очистить нашу историю\n"
        "/health — проверить статус сервисов\n"
        "/add_to_vector — добавляет данные в базу знаний"
    )


@dp.message(Command("health"))
async def health_command(message: Message):
    await message.answer(
        "🩺 Состояние системы:\n"
        "• Память чатов: подключена ✅\n"
        "• Векторное хранилище (ChromaDB): активно ✅\n"
        "• LLM (Ollama): доступна ✅\n"
        "• MCP инструменты: погода, поиск в интернете, Notion — готовы к работе ✅\n\n"
        "Если что-то не отвечает, дай знать — проверю детали."
    )


@dp.message(Command("new_chat"))
async def new_chat_command(message: Message):
    memory.clear_user_memory(message.from_user.id)
    await message.answer("История сообщений очищена.")


@dp.message(Command("add_to_vector"))
async def add_to_vector_start(message: Message, state: FSMContext) -> None:
    """
    Запускает сценарий добавления текста в векторную базу.
    """
    await state.set_state(AddToVectorStates.waiting_for_text)
    await message.answer(
        "Отправь текст, который нужно сохранить в векторную базу.\n\n"
        "Если передумал — отправь /cancel."
    )


@dp.message(Command("cancel"))
async def cancel_fsm(message: Message, state: FSMContext) -> None:
    """
    Универсальная команда для выхода из любого состояния FSM.
    """
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Сейчас никакой активной операции нет.")
        return

    await state.clear()
    await message.answer("Текущая операция отменена.")


@dp.message(AddToVectorStates.waiting_for_text, F.text)
async def add_to_vector_save(message: Message, state: FSMContext) -> None:
    """
    Обрабатывает текст от пользователя и сохраняет его в векторную базу.
    """
    user_text = (message.text or "").strip()
    if not user_text:
        await message.answer("Текст пустой. Пожалуйста, отправь непустое сообщение.")
        return

    # Генерируем уникальный идентификатор документа
    doc_id = str(uuid.uuid4())

    # Сохраняем текст в векторное хранилище (асинхронно, чтобы не блокировать event loop)
    await vector_db.add_async(ids=[doc_id], documents=[user_text])

    await state.clear()
    await message.answer(
        "Текст успешно сохранён в векторную базу.\n"
        "ID сохранённого документа:\n"
        f"`{doc_id}`"
    )


@dp.message(F.text)
async def text_handler(message: Message):
    user_id = message.from_user.id
    user_text = message.text

    # сохраняем сообщение пользователя
    memory.add_message(user_id, "user", user_text)

    # 1. ПОИСК ПО EMBEDDINGS (асинхронно, чтобы не блокировать event loop)
    results = await vector_db.query_async(user_text, n_results=3)
    similarity_threshold = 0.7
    best_answer = None
    best_score = 0

    if results and "documents" in results and "distances" in results:
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            sim = 1 - dist

            if sim > best_score:
                best_score = sim
                best_answer = doc

    # 2. Если документ достаточно похож → используем его
    if best_score >= similarity_threshold:
        answer = best_answer
    else:
        # 3. Иначе — вызываем LLM Ollama
        chat_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *memory.get_conversation_context(user_id, SYSTEM_PROMPT),
        ]
        llm_response = await llm_ollama(chat_messages)

        # Проверяем, что LLM вернула текст
        if llm_response and llm_response.strip():
            answer = llm_response
        elif best_answer:
            answer = best_answer
        else:
            answer = "Извини, я пока не могу дать ответ."

    # 4. сохраняем ответ
    memory.add_message(user_id, "assistant", answer)

    # 5. отправка пользователю
    await message.answer(answer)


@dp.message()
async def not_text(message: Message):
    await message.answer("Я работаю только с текстом.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

```

### Что сделал

- В `vector_store.py` уже используется `base_url="http://ollama:11434"` для `OllamaEmbeddings` — там ничего менять не нужно.
- В `bot.py` добавил явный клиент Ollama с нужным хостом и перевёл все вызовы на него:

```startLine:endLine:bot.py
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Явно указываем base_url для доступа к сервису Ollama внутри docker-compose
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
```

```startLine:endLine:bot.py
async def call_ollama(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Запускает обращение к Ollama в отдельном потоке, чтобы не блокировать event-loop."""
    return await asyncio.to_thread(
        lambda: ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
    )
```

Теперь бот **всегда** ходит к Ollama по `http://ollama:11434` (можно переопределить через переменную окружения `OLLAMA_BASE_URL` при необходимости).

```python
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Явно указываем base_url для доступа к сервису Ollama внутри docker-compose
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
```

```python
async def call_ollama(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Запускает обращение к Ollama в отдельном потоке, чтобы не блокировать event-loop."""
    return await asyncio.to_thread(
        lambda: ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
    )
```

