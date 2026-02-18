import asyncio
from pathlib import Path

import pytest

from services import mapy_cz


@pytest.mark.asyncio
async def test_elevation_profile_cache(tmp_path, monkeypatch):
    async def fake_elevation(positions, lang="en"):
        return [{"position": {"lon": lon, "lat": lat}, "elevation": idx * 10} for idx, (lon, lat) in enumerate(positions)]

    monkeypatch.setattr(mapy_cz, "ELEVATION_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mapy_cz, "elevation_for_positions", fake_elevation)

    coordinates = [[14.0, 50.0], [14.1, 50.1], [14.2, 50.2]]
    profile = await mapy_cz.elevation_profile_for_route(coordinates)

    assert len(profile) == 3
    assert profile[0]["elevation"] == 0
    assert profile[1]["elevation"] == 10

    # Cache should be reused
    profile_cached = await mapy_cz.elevation_profile_for_route(coordinates)
    assert profile_cached == profile


def test_build_mapy_url_showmap():
    url = mapy_cz.build_mapy_url_showmap([14.42, 50.08], zoom=15, mapset="outdoor")
    assert "showmap" in url
    assert "center=14.42%2C50.08" in url


def test_parse_mapy_route_url():
    url = "https://mapy.com/fnc/v1/route?start=14.42,50.08&end=14.5,50.1&routeType=foot_hiking&waypoints=14.43,50.081;14.44,50.09"
    result = mapy_cz.parse_mapy_url(url)
    assert result["type"] == "route"
    assert result["start"] == [14.42, 50.08]
    assert result["end"] == [14.5, 50.1]
    assert len(result["waypoints"]) == 2
