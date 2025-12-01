import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

app = FastMCP()


class WeatherInput(BaseModel):
    city: str


class WeatherOutput(BaseModel):
    city: str
    temperature: float
    windspeed: float
    weathercode: int
    latitude: float
    longitude: float


@app.tool(name="get_weather", description="Получает текущую погоду по названию города")
def get_weather(data: WeatherInput) -> WeatherOutput:
    city = data.city

    # 1) Получаем координаты города
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={city}&count=1&language=ru&format=json"
    )
    geo_resp = requests.get(geo_url).json()

    if "results" not in geo_resp:
        raise Exception("Город не найден!")

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]

    # 2) Получаем погоду по координатам
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?current_weather=true&latitude={lat}&longitude={lon}"
    )
    weather_resp = requests.get(weather_url).json()

    weather_data = weather_resp["current_weather"]

    # 3) Возвращаем результат
    return WeatherOutput(
        city=city,
        temperature=weather_data["temperature"],
        windspeed=weather_data["windspeed"],
        weathercode=weather_data["weathercode"],
        latitude=lat,
        longitude=lon,
    )


if __name__ == "__main__":
    print("MCP погода запущен")
    app.run()
