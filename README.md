# Hike Guide

Hike Guide is a FastAPI app for tracking hikes, uploading GPX routes, and enriching routes with Mapy.com data (tiles, routing, geocoding, elevation, and static maps).

## Features

- GPX upload with route rendering.
- Mapy.com route planning and snapping.
- Elevation profiles via Mapy.com Elevation API.
- Static map thumbnails for hike cards.
- Mapy.com deep links for maps and routes.

## Configuration

Create a `.env` file (see `.env.example`) and set your Mapy.com API key:

````text
MAPY_CZ_API_KEY=your_api_key_here
````

Optional settings are documented in `.env.example`.

## Run locally

````bash
python -m venv .venv
````

````bash
.\.venv\Scripts\activate
````

````bash
pip install -r requirements.txt
````

````bash
python main.py
````

Then open `http://localhost:8000`.

## Tests

````bash
pytest
````

## Notes

- The Mapy.com REST API uses the `MAPY_CZ_API_KEY` value; keep it out of source control.
- If Mapy tiles do not load, update `MAPY_CZ_TILE_URL_TEMPLATE` in `.env` to match the latest Mapy.com tiles endpoint.
