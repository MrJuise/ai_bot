from typing import List

from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

app = FastMCP()


class DuckDuckGoInput(BaseModel):
    query: str = Field(..., description="Поисковый запрос пользователя")


class DuckDuckGoResult(BaseModel):
    title: str
    snippet: str
    url: str


class DuckDuckGoOutput(BaseModel):
    answer: str
    results: List[DuckDuckGoResult]


def _format_result(item: dict) -> DuckDuckGoResult:
    return DuckDuckGoResult(
        title=item.get("title") or "Результат",
        snippet=item.get("body") or "",
        url=item.get("href") or "",
    )


@app.tool(name="duckduckgo_search", description="Поиск в DuckDuckGo")
def duckduckgo_search(data: DuckDuckGoInput) -> DuckDuckGoOutput:
    with DDGS() as ddgs:
        results_raw = list(ddgs.text(data.query, max_results=5))

    results = [_format_result(item) for item in results_raw]

    if not results:
        results.append(
            DuckDuckGoResult(
                title="Ничего не найдено",
                snippet="DuckDuckGo не вернул результатов по запросу.",
                url="",
            )
        )

    return DuckDuckGoOutput(
        answer=f"Нашёл {len(results)} результат(ов).", results=results[:5]
    )
