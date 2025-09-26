import requests
from ..core.config import configs


def get_region_dadata(city_name):
    url = "https://suggestions.dadata.ru/suggestions/api/4_1/rs/suggest/address"
    headers = {
        "Authorization": f"Token {configs.GEO_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {"query": city_name}
    response = requests.post(url, headers=headers, json=data)
    if response.json()["suggestions"]:
        suggestion = response.json()["suggestions"][0]
        return suggestion["data"].get("region_iso_code", None)
    return None
