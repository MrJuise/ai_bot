import os
from typing import List, Optional
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

load_dotenv()

app = FastMCP(stateless_http=True, json_response=True)

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_VERSION = "2025-09-03"
NOTION_BASE_URL = "https://api.notion.com/v1"
PARENT_ID = os.getenv("PARENT_ID")


class NotionSearchInput(BaseModel):
    query: str = Field(..., description="Поисковый запрос для поиска страниц в Notion")


class NotionPageResult(BaseModel):
    id: str
    title: str
    url: str
    last_edited_time: str


class NotionSearchOutput(BaseModel):
    results: List[NotionPageResult]
    total: int


class NotionGetPageInput(BaseModel):
    page_id: str = Field(..., description="ID страницы Notion для получения содержимого")


class NotionPageContent(BaseModel):
    id: str
    title: str
    url: str
    content: str
    properties: dict


class NotionCreatePageInput(BaseModel):
    parent_id: Optional[str] = Field(None, description="ID родительской страницы или базы данных (если не указан, используется PARENT_ID из .env)")
    title: str = Field(..., description="Заголовок новой страницы")
    content: Optional[str] = Field(None, description="Содержимое страницы (текст)")


class NotionCreatePageOutput(BaseModel):
    id: str
    title: str
    url: str
    created: bool


def _get_headers() -> dict:
    """Возвращает заголовки для запросов к Notion API."""
    if not NOTION_API_KEY:
        raise ValueError("NOTION_API_KEY не установлен в переменных окружения")
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _extract_title_from_page(page: dict) -> str:
    """Извлекает заголовок страницы из объекта страницы."""
    properties = page.get("properties", {})
    for prop_name, prop_value in properties.items():
        if prop_value.get("type") == "title":
            title_array = prop_value.get("title", [])
            if title_array:
                return title_array[0].get("text", {}).get("content", "Без названия")
    return "Без названия"


@app.tool(
    name="notion_search",
    description="Поиск страниц в Notion по запросу",
)
def notion_search(data: NotionSearchInput) -> NotionSearchOutput:
    """Ищет страницы в Notion по текстовому запросу."""
    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            f"{NOTION_BASE_URL}/search",
            headers=_get_headers(),
            json={"query": data.query, "page_size": 10},
        )
        response.raise_for_status()
        search_data = response.json()

    results = []
    for page in search_data.get("results", []):
        if page.get("object") == "page":
            results.append(
                NotionPageResult(
                    id=page["id"],
                    title=_extract_title_from_page(page),
                    url=page.get("url", ""),
                    last_edited_time=page.get("last_edited_time", ""),
                )
            )

    return NotionSearchOutput(results=results, total=len(results))


@app.tool(
    name="notion_get_page",
    description="Получает содержимое страницы Notion по ID",
)
def notion_get_page(data: NotionGetPageInput) -> NotionPageContent:
    """Получает содержимое страницы Notion."""
    with httpx.Client(timeout=10.0) as client:
        # Получаем страницу
        page_response = client.get(
            f"{NOTION_BASE_URL}/pages/{data.page_id}",
            headers=_get_headers(),
        )
        page_response.raise_for_status()
        page_data = page_response.json()

        # Получаем блоки страницы
        blocks_response = client.get(
            f"{NOTION_BASE_URL}/blocks/{data.page_id}/children",
            headers=_get_headers(),
        )
        blocks_response.raise_for_status()
        blocks_data = blocks_response.json()

    # Извлекаем текст из блоков
    content_parts = []
    for block in blocks_data.get("results", []):
        block_type = block.get("type")
        if block_type == "paragraph":
            rich_text = block.get("paragraph", {}).get("rich_text", [])
            for text_obj in rich_text:
                content_parts.append(text_obj.get("plain_text", ""))
        elif block_type == "heading_1":
            rich_text = block.get("heading_1", {}).get("rich_text", [])
            for text_obj in rich_text:
                content_parts.append(f"# {text_obj.get('plain_text', '')}")
        elif block_type == "heading_2":
            rich_text = block.get("heading_2", {}).get("rich_text", [])
            for text_obj in rich_text:
                content_parts.append(f"## {text_obj.get('plain_text', '')}")
        elif block_type == "heading_3":
            rich_text = block.get("heading_3", {}).get("rich_text", [])
            for text_obj in rich_text:
                content_parts.append(f"### {text_obj.get('plain_text', '')}")

    content = "\n".join(content_parts) or "Страница пуста"

    return NotionPageContent(
        id=page_data["id"],
        title=_extract_title_from_page(page_data),
        url=page_data.get("url", ""),
        content=content,
        properties=page_data.get("properties", {}),
    )


@app.tool(
    name="notion_create_page",
    description="Создаёт новую страницу в Notion. Если parent_id не указан, используется PARENT_ID из .env",
)
def notion_create_page(data: NotionCreatePageInput) -> NotionCreatePageOutput:
    """Создаёт новую страницу в Notion."""
    # Используем parent_id из аргументов или PARENT_ID из .env
    parent_id = data.parent_id or PARENT_ID
    if not parent_id:
        raise ValueError(
            "parent_id не указан в запросе и PARENT_ID не установлен в переменных окружения. "
            "Укажите parent_id или добавьте PARENT_ID в .env файл."
        )

    # Определяем тип родителя: пробуем сначала page_id, потом database_id
    # UUID страницы обычно имеет формат с дефисами (36 символов) или без (32 символа)
    parent_id_clean = parent_id.replace("-", "")
    
    # Пробуем сначала как page_id (обычно это UUID)
    parent_key = "page_id"
    if len(parent_id_clean) != 32:
        # Если не похоже на UUID, пробуем как database_id
        parent_key = "database_id"

    request_body = {
        "parent": {parent_key: parent_id},
        "properties": {
            "title": {
                "title": [
                    {
                        "text": {
                            "content": data.title,
                        }
                    }
                ]
            }
        },
    }

    # Добавляем содержимое, если указано
    children = []
    if data.content:
        children.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": data.content,
                            },
                        }
                    ],
                },
            }
        )
        request_body["children"] = children

    with httpx.Client(timeout=10.0) as client:
        try:
            response = client.post(
                f"{NOTION_BASE_URL}/pages",
                headers=_get_headers(),
                json=request_body,
            )
            response.raise_for_status()
            page_data = response.json()
        except httpx.HTTPStatusError as e:
            error_detail = ""
            if e.response.status_code == 404:
                error_detail = (
                    f"Страница или база данных с ID '{parent_id}' не найдена. "
                    "Проверьте, что:\n"
                    "1. ID указан правильно\n"
                    "2. Страница/база данных доступна для вашего интеграционного ключа\n"
                    "3. Интеграция имеет права на создание страниц в этом родителе"
                )
            elif e.response.status_code == 401:
                error_detail = "Ошибка авторизации. Проверьте правильность NOTION_API_KEY"
            elif e.response.status_code == 403:
                error_detail = "Доступ запрещён. Убедитесь, что интеграция имеет необходимые права"
            else:
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("message", str(e))
                except (ValueError, KeyError):
                    error_detail = str(e)
            
            raise ValueError(f"Ошибка при создании страницы в Notion: {error_detail}")

    return NotionCreatePageOutput(
        id=page_data["id"],
        title=_extract_title_from_page(page_data),
        url=page_data.get("url", ""),
        created=True,
    )


if __name__ == "__main__":
    print("MCP Notion запущен с потоковым HTTP")
    app.run(transport="streamable-http")
