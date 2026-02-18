from __future__ import annotations

import hashlib
import json
import math
import os
import struct as _struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

import httpx

API_KEY_ENV = "MAPY_CZ_API_KEY"
DEFAULT_BASE_URL = os.getenv("MAPY_CZ_BASE_URL", "https://api.mapy.com")
DEFAULT_MAPSET = os.getenv("MAPY_CZ_MAPSET", "outdoor")
DEFAULT_ROUTE_TYPE = os.getenv("MAPY_CZ_ROUTE_TYPE", "foot_hiking")
DEFAULT_LANG = os.getenv("MAPY_CZ_LANG", "en")

DEFAULT_TILE_URL_TEMPLATE = os.getenv(
    "MAPY_CZ_TILE_URL_TEMPLATE",
    "https://api.mapy.com/v1/maptiles/{mapset}/256/{z}/{x}/{y}",
)
TILE_JSON_URL_TEMPLATE = os.getenv(
    "MAPY_CZ_TILE_JSON_URL_TEMPLATE",
    "https://api.mapy.com/v1/maptiles/{mapset}/tiles.json",
)

CACHE_ROOT = Path(os.getenv("MAPY_CZ_CACHE_DIR", "cache"))
ELEVATION_CACHE_DIR = CACHE_ROOT / "mapy_elevation"
STATIC_MAP_CACHE_DIR = CACHE_ROOT / "mapy_static_map"

MAPY_URL_BASE = "https://mapy.com/fnc/v1"

MAPSET_ALIASES = {
    "basic": "basic",
    "base": "basic",
    "outdoor": "outdoor",
    "tourist": "outdoor",
    "turist": "outdoor",
    "turist-m": "outdoor",
    "winter": "winter",
    "aerial": "aerial",
    "names-overlay": "names-overlay",
    "aerial-names-overlay": "names-overlay",
    "labels": "names-overlay",
    "traffic": "basic",
}


@dataclass
class MapyRouteResult:
    length_meters: Optional[int]
    duration_seconds: Optional[int]
    geometry: Optional[dict]
    route_points: Optional[list]


class MapyApiError(RuntimeError):
    pass


def get_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise MapyApiError("MAPY_CZ_API_KEY is not configured")
    return api_key


def _ensure_cache_dirs() -> None:
    ELEVATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_positions(positions: Sequence[Sequence[float]]) -> list[str]:
    return [f"{lon},{lat}" for lon, lat in positions]


def _sample_coordinates(coordinates: Sequence[Sequence[float]], limit: int = 256) -> list[list[float]]:
    if len(coordinates) <= limit:
        return [list(coord) for coord in coordinates]
    step = max(1, math.floor(len(coordinates) / limit))
    sampled = [coordinates[idx] for idx in range(0, len(coordinates), step)]
    return [list(coord) for coord in sampled[:limit]]


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


# ---------------------------------------------------------------------------
# FRPC binary protocol (Seznam proprietary RPC used by mapy.com)
# ---------------------------------------------------------------------------
_FRPC_TYPE_INT = 1
_FRPC_TYPE_BOOL = 2
_FRPC_TYPE_DOUBLE = 3
_FRPC_TYPE_STRING = 4
_FRPC_TYPE_DATETIME = 5
_FRPC_TYPE_BINARY = 6
_FRPC_TYPE_INT8P = 7
_FRPC_TYPE_INT8N = 8
_FRPC_TYPE_STRUCT = 10
_FRPC_TYPE_ARRAY = 11
_FRPC_TYPE_NULL = 12
_FRPC_TYPE_CALL = 13
_FRPC_TYPE_RESPONSE = 14
_FRPC_TYPE_FAULT = 15


def _frpc_encode_int(value: int) -> list[int]:
    if not value:
        return [0]
    result = []
    remain = value
    while remain:
        result.append(remain % 256)
        remain = (remain - result[-1]) // 256
    return result


def _frpc_serialize_value(result: list[int], value) -> None:
    if value is None:
        result.append(_FRPC_TYPE_NULL << 3)
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        int_data = _frpc_encode_int(len(encoded))
        result.append((_FRPC_TYPE_STRING << 3) + len(int_data) - 1)
        result.extend(int_data)
        result.extend(encoded)
    elif isinstance(value, bool):
        result.append((_FRPC_TYPE_BOOL << 3) + (1 if value else 0))
    elif isinstance(value, int):
        typ = _FRPC_TYPE_INT8P if value >= 0 else _FRPC_TYPE_INT8N
        data = _frpc_encode_int(abs(value))
        result.append((typ << 3) + len(data) - 1)
        result.extend(data)
    elif isinstance(value, float):
        result.append(_FRPC_TYPE_DOUBLE << 3)
        result.extend(_struct.pack("<d", value))
    elif isinstance(value, list):
        int_data = _frpc_encode_int(len(value))
        result.append((_FRPC_TYPE_ARRAY << 3) + len(int_data) - 1)
        result.extend(int_data)
        for item in value:
            _frpc_serialize_value(result, item)
    elif isinstance(value, dict):
        int_data = _frpc_encode_int(len(value))
        result.append((_FRPC_TYPE_STRUCT << 3) + len(int_data) - 1)
        result.extend(int_data)
        for k, v in value.items():
            kb = k.encode("utf-8")
            result.append(len(kb))
            result.extend(kb)
            _frpc_serialize_value(result, v)


def _frpc_serialize_call(method: str, args: list) -> bytes:
    """Build an FRPC binary call (version 2.1)."""
    body: list[int] = []
    _frpc_serialize_value(body, args)  # serialized as array
    body = body[2:]  # strip array header (type byte + length byte)
    method_bytes = method.encode("utf-8")
    header = [0xCA, 0x11, 0x02, 0x01, _FRPC_TYPE_CALL << 3, len(method_bytes)]
    return bytes(header + list(method_bytes) + body)


class _FRPCParser:
    """Minimal FRPC binary response parser."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def _byte(self) -> int:
        b = self._data[self._pos]
        self._pos += 1
        return b

    def _int(self, n: int) -> int:
        val = 0
        for i in range(n):
            val += self._data[self._pos + i] << (8 * i)
        self._pos += n
        return val

    def _bytes(self, n: int) -> bytes:
        result = self._data[self._pos : self._pos + n]
        self._pos += n
        return result

    def _value(self):
        first = self._byte()
        tid = first >> 3
        if tid == _FRPC_TYPE_STRING:
            length = self._int((first & 7) + 1)
            return self._bytes(length).decode("utf-8", errors="replace")
        if tid == _FRPC_TYPE_STRUCT:
            members = self._int((first & 7) + 1)
            result = {}
            for _ in range(members):
                kl = self._byte()
                key = self._bytes(kl).decode("utf-8", errors="replace")
                result[key] = self._value()
            return result
        if tid == _FRPC_TYPE_ARRAY:
            count = self._int((first & 7) + 1)
            return [self._value() for _ in range(count)]
        if tid == _FRPC_TYPE_INT8P:
            return self._int((first & 7) + 1)
        if tid == _FRPC_TYPE_INT8N:
            return -self._int((first & 7) + 1)
        if tid == _FRPC_TYPE_INT:
            return self._int((first & 7) + 1)
        if tid == _FRPC_TYPE_BOOL:
            return bool(first & 7)
        if tid == _FRPC_TYPE_DOUBLE:
            return _struct.unpack("<d", self._bytes(8))[0]
        if tid == _FRPC_TYPE_NULL:
            return None
        if tid == _FRPC_TYPE_BINARY:
            length = self._int((first & 7) + 1)
            return self._bytes(length)
        if tid == _FRPC_TYPE_DATETIME:
            self._bytes(10)  # skip datetime payload
            return None
        raise ValueError(f"Unknown FRPC type {tid}")

    def parse(self):
        m1, m2 = self._byte(), self._byte()
        if m1 != 0xCA or m2 != 0x11:
            raise ValueError("Invalid FRPC magic")
        self._pos += 2  # skip version bytes
        first = self._byte()
        tid = first >> 3
        if tid == _FRPC_TYPE_RESPONSE:
            return self._value()
        if tid == _FRPC_TYPE_FAULT:
            return {"__fault__": True, "data": self._value()}
        raise ValueError(f"Expected RESPONSE/FAULT, got type {tid}")


# ---------------------------------------------------------------------------
# Mapy.com geometry string decoder (SMap.Coords.stringToCoords)
# ---------------------------------------------------------------------------
_GEOM_ALPHABET = "0ABCD2EFGH4IJKLMN6OPQRST8UVWXYZ-1abcd3efgh5ijklmn7opqrst9uvwxyz."


def _geom_parse_number(arr: list[str], count: int) -> int:
    result = 0
    remaining = count
    while remaining:
        if not arr:
            raise ValueError("Geometry decode: no data")
        ch = arr.pop()
        idx = _GEOM_ALPHABET.index(ch)
        if idx == -1:
            continue
        result = (result << 6) + idx
        remaining -= 1
    return result


def decode_geometry_string(s: str) -> list[tuple[float, float]]:
    """Decode a mapy.com geometry string to [(lon, lat), ...]."""
    FIVE = (1 + 2) << 4  # 48
    THREE = 1 << 5  # 32
    results: list[tuple[float, float]] = []
    coords = [0, 0]
    ci = 0
    arr = list(s.strip())
    arr.reverse()
    while arr:
        num = _geom_parse_number(arr, 1)
        if (num & FIVE) == FIVE:
            num -= FIVE
            num = ((num & 15) << 24) + _geom_parse_number(arr, 4)
            coords[ci] = num
        elif (num & THREE) == THREE:
            num = ((num & 15) << 12) + _geom_parse_number(arr, 2)
            num -= 1 << 15
            coords[ci] += num
        else:
            num = ((num & 31) << 6) + _geom_parse_number(arr, 1)
            num -= 1 << 10
            coords[ci] += num
        if ci:
            x = coords[0] * 360 / (1 << 28) - 180
            y = coords[1] * 180 / (1 << 28) - 90
            results.append((x, y))  # (lon, lat)
        ci = (ci + 1) % 2
    return results


# ---------------------------------------------------------------------------
# Resolve mapy.com 'dim' shared items via MapyBox FRPC API
# ---------------------------------------------------------------------------
_MAPYBOX_URL = "https://mapy.com/api/mapybox-ng/"


async def resolve_dim(dim_id: str) -> Optional[dict]:
    """Call like.detail via FRPC to resolve a mapy.com shared item (dim).

    Returns a dict with keys like: title, type, start, end, waypoints,
    route_geometry (list of [lon, lat]), totalLength, totalTime, routeType,
    mapset, bbox, etc.  Returns None on failure.
    """
    call_bytes = _frpc_serialize_call("like.detail", [dim_id, {"lang": ["en", "cs"]}])
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _MAPYBOX_URL,
                content=call_bytes,
                headers={
                    "Content-Type": "application/x-frpc",
                    "Accept": "application/x-frpc",
                },
            )
        if resp.status_code != 200:
            return None
        parsed = _FRPCParser(resp.content).parse()
    except Exception as exc:
        print(f"[dim] FRPC call failed: {exc}")
        return None

    like = parsed.get("like") if isinstance(parsed, dict) else None
    if not like:
        return None

    result: dict = {
        "title": like.get("usertitle") or like.get("title") or "",
        "dim_type": like.get("type", ""),
    }

    # Extract bounding box
    bbox = like.get("bbox")
    if bbox and len(bbox) == 4:
        result["bbox"] = bbox  # [min_lon, min_lat, max_lon, max_lat]

    data = like.get("data", {})
    route_points = data.get("route", [])

    # Decode waypoint coordinates from their geometry strings
    waypoints: list[list[float]] = []
    for rp in route_points:
        geom_str = rp.get("geometry", "")
        if geom_str:
            try:
                pts = decode_geometry_string(geom_str)
                if pts:
                    waypoints.append([pts[0][0], pts[0][1]])  # [lon, lat]
            except Exception:
                pass

    if waypoints:
        result["start"] = waypoints[0]
        result["end"] = waypoints[-1]
        if len(waypoints) > 2:
            result["waypoints"] = waypoints[1:-1]

    # Decode full route polyline
    full_geom = data.get("geometry", "")
    if full_geom:
        try:
            coords = decode_geometry_string(full_geom)
            if coords:
                result["route_geometry"] = [[lon, lat] for lon, lat in coords]
        except Exception:
            pass

    # Length and time
    if data.get("totalLength"):
        result["totalLength"] = data["totalLength"]
    if data.get("totalTime"):
        result["totalTime"] = data["totalTime"]

    # Route type from the first waypoint's routeParams
    if route_points:
        rp0 = route_points[0].get("routeParams", {})
        criterion = rp0.get("criterion")
        if criterion is not None:
            result["routeType"] = _MRP_ROUTE_TYPES.get(int(criterion), DEFAULT_ROUTE_TYPE)

    # Fallback center from mark
    mark = like.get("mark")
    if mark:
        result["center"] = [mark.get("lon", 0), mark.get("lat", 0)]

    return result


def build_elevation_profile(
    coordinates: Sequence[Sequence[float]], elevations: Sequence[float]
) -> list[dict]:
    profile = []
    distance = 0.0
    prev_lat = None
    prev_lon = None
    for (lon, lat), elevation in zip(coordinates, elevations):
        if prev_lat is not None:
            distance += _haversine_distance(prev_lat, prev_lon, lat, lon)
        profile.append({"distance": distance, "elevation": elevation, "lon": lon, "lat": lat})
        prev_lat, prev_lon = lat, lon
    return profile


async def _request_json(path: str, params: dict, base_url: str = DEFAULT_BASE_URL) -> dict:
    api_key = get_api_key()
    headers = {"X-Mapy-Api-Key": api_key}
    async with httpx.AsyncClient(base_url=base_url, timeout=20.0) as client:
        response = await client.get(path, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


async def geocode(query: str, limit: int = 5, lang: str = DEFAULT_LANG) -> dict:
    return await _request_json(
        "/v1/geocode",
        {"query": query, "limit": limit, "lang": lang},
    )


async def suggest(query: str, limit: int = 5, lang: str = DEFAULT_LANG) -> dict:
    return await _request_json(
        "/v1/suggest",
        {"query": query, "limit": limit, "lang": lang},
    )


async def reverse_geocode(lon: float, lat: float, lang: str = DEFAULT_LANG) -> dict:
    return await _request_json(
        "/v1/rgeocode",
        {"lon": lon, "lat": lat, "lang": lang},
    )


async def plan_route(
    start: Sequence[float],
    end: Sequence[float],
    route_type: str = DEFAULT_ROUTE_TYPE,
    waypoints: Optional[Sequence[Sequence[float]]] = None,
    lang: str = DEFAULT_LANG,
    geometry_format: str = "geojson",
) -> MapyRouteResult:
    params = {
        "start": f"{start[0]},{start[1]}",
        "end": f"{end[0]},{end[1]}",
        "routeType": route_type,
        "lang": lang,
        "format": geometry_format,
    }
    if waypoints:
        params["waypoints"] = [f"{lon},{lat}" for lon, lat in waypoints]
    data = await _request_json("/v1/routing/route", params)
    return MapyRouteResult(
        length_meters=data.get("length"),
        duration_seconds=data.get("duration"),
        geometry=data.get("geometry"),
        route_points=data.get("routePoints"),
    )


async def elevation_for_positions(
    positions: Sequence[Sequence[float]], lang: str = DEFAULT_LANG
) -> list[dict]:
    if not positions:
        return []
    params = {
        "positions": _format_positions(positions),
        "lang": lang,
    }
    data = await _request_json("/v1/elevation", params)
    return data.get("items", [])


async def elevation_profile_for_route(
    coordinates: Sequence[Sequence[float]],
    lang: str = DEFAULT_LANG,
) -> list[dict]:
    if not coordinates:
        return []
    _ensure_cache_dirs()
    sampled = _sample_coordinates(coordinates)
    cache_key = _hash_payload(json.dumps(sampled))
    cache_path = ELEVATION_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    items = await elevation_for_positions(sampled, lang=lang)
    elevations = [item.get("elevation") for item in items]
    profile = build_elevation_profile(sampled, elevations)
    cache_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile


async def get_static_map_image(
    lon_lat_bounds: Optional[Sequence[Sequence[float]]] = None,
    markers: Optional[Sequence[str]] = None,
    shapes: Optional[Sequence[str]] = None,
    width: int = 600,
    height: int = 400,
    mapset: str = DEFAULT_MAPSET,
    image_format: str = "png",
    lang: str = DEFAULT_LANG,
) -> tuple[bytes, str]:
    _ensure_cache_dirs()
    payload = json.dumps(
        {
            "bounds": lon_lat_bounds,
            "markers": markers,
            "shapes": shapes,
            "width": width,
            "height": height,
            "mapset": mapset,
            "format": image_format,
            "lang": lang,
        },
        sort_keys=True,
    )
    cache_key = _hash_payload(payload)
    cache_path = STATIC_MAP_CACHE_DIR / f"{cache_key}.{image_format}"
    if cache_path.exists():
        return cache_path.read_bytes(), f"image/{image_format}"

    params = {
        "width": width,
        "height": height,
        "mapset": mapset,
        "format": image_format,
        "lang": lang,
    }
    if lon_lat_bounds:
        lons = [coord[0] for coord in lon_lat_bounds]
        lats = [coord[1] for coord in lon_lat_bounds]
        params["lon"] = lons
        params["lat"] = lats
    if markers:
        params["markers"] = list(markers)
    if shapes:
        params["shapes"] = list(shapes)

    api_key = get_api_key()
    headers = {"X-Mapy-Api-Key": api_key}
    async with httpx.AsyncClient(base_url=DEFAULT_BASE_URL, timeout=20.0) as client:
        response = await client.get("/v1/static/map", params=params, headers=headers)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
        content_type = response.headers.get("Content-Type", f"image/{image_format}")
        return response.content, content_type


def build_mapy_url_showmap(
    center: Sequence[float],
    zoom: int = 13,
    mapset: str = DEFAULT_MAPSET,
    marker: bool = True,
) -> str:
    mapset = normalize_mapset(mapset)
    params = {
        "center": f"{center[0]},{center[1]}",
        "zoom": zoom,
        "mapset": mapset,
        "marker": str(marker).lower(),
    }
    return f"{MAPY_URL_BASE}/showmap?{urlencode(params)}"


def build_mapy_url_route(
    start: Sequence[float],
    end: Sequence[float],
    route_type: str = DEFAULT_ROUTE_TYPE,
    waypoints: Optional[Sequence[Sequence[float]]] = None,
    mapset: str = DEFAULT_MAPSET,
) -> str:
    mapset = normalize_mapset(mapset)
    params = {
        "start": f"{start[0]},{start[1]}",
        "end": f"{end[0]},{end[1]}",
        "routeType": route_type,
        "mapset": mapset,
    }
    if waypoints:
        params["waypoints"] = ";".join([f"{lon},{lat}" for lon, lat in waypoints])
    return f"{MAPY_URL_BASE}/route?{urlencode(params)}"


def parse_mapy_url(url: str) -> dict:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    path = parsed.path
    result: dict = {"raw": url, "path": path}
    if path.endswith("/showmap"):
        center = query.get("center", [None])[0]
        if center:
            lon, lat = center.split(",")
            result["center"] = [float(lon), float(lat)]
        result["mapset"] = normalize_mapset(query.get("mapset", [DEFAULT_MAPSET])[0])
        result["zoom"] = query.get("zoom", [None])[0]
        result["type"] = "showmap"
    elif path.endswith("/route"):
        start = query.get("start", [None])[0]
        end = query.get("end", [None])[0]
        if start:
            lon, lat = start.split(",")
            result["start"] = [float(lon), float(lat)]
        if end:
            lon, lat = end.split(",")
            result["end"] = [float(lon), float(lat)]
        waypoints = query.get("waypoints", [None])[0]
        if waypoints:
            wp_list = []
            for point in waypoints.split(";"):
                lon, lat = point.split(",")
                wp_list.append([float(lon), float(lat)])
            result["waypoints"] = wp_list
        result["routeType"] = query.get("routeType", [DEFAULT_ROUTE_TYPE])[0]
        result["mapset"] = normalize_mapset(query.get("mapset", [DEFAULT_MAPSET])[0])
        result["type"] = "route"
    return result


# Mapping of mapy.com mrp "c" values to route types
_MRP_ROUTE_TYPES: dict[int, str] = {
    1: "car_fast",
    2: "car_short",
    3: "car_fast_traffic",
    4: "foot_fast",
    5: "foot_hiking",      # pěší turistika
    6: "bike_road",
    7: "bike_mountain",
    128: "car_fast",
    129: "car_short",
    130: "car_fast_traffic",
    131: "foot_fast",
    132: "foot_hiking",
    133: "bike_road",
    134: "bike_mountain",
}

# Mapping of mapy.com path segments to mapset names
_PATH_MAPSETS: dict[str, str] = {
    "zakladni": "basic",
    "turisticka": "outdoor",
    "letecka": "aerial",
    "zimni": "winter",
    "dopravni": "basic",
    "base": "basic",
    "outdoor": "outdoor",
    "aerial": "aerial",
    "winter": "winter",
    "traffic": "basic",
}


def _parse_legacy_mapy_url(url: str) -> dict:
    """Parse old-format mapy.com URLs (e.g. from shared links / short URLs).

    Example resolved URL:
    https://mapy.com/en/turisticka?planovani-trasy&rc=...&x=14.41&y=50.05&z=12&mrp={"c":132}
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    result: dict = {"raw": url}

    # Detect mapset from path segment (e.g. /en/turisticka or /turisticka)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    for part in path_parts:
        if part in _PATH_MAPSETS:
            result["mapset"] = _PATH_MAPSETS[part]
            break

    # x/y/z are the map center and zoom
    x_val = query.get("x", [None])[0]
    y_val = query.get("y", [None])[0]
    z_val = query.get("z", [None])[0]
    if x_val and y_val:
        try:
            result["center"] = [float(x_val), float(y_val)]
        except ValueError:
            pass
    if z_val:
        result["zoom"] = z_val

    # Extract dim parameter (shared item ID for MapyBox)
    dim_val = query.get("dim", [None])[0]
    if dim_val:
        result["dim"] = dim_val

    # Detect route via "planovani-trasy" key in query string
    is_route = "planovani-trasy" in query or "planovani-trasy" in parsed.query

    if is_route:
        result["type"] = "route"

        # Try to extract route type from mrp JSON
        mrp_raw = query.get("mrp", [None])[0]
        if mrp_raw:
            try:
                import json as _json
                mrp = _json.loads(mrp_raw)
                c_val = mrp.get("c")
                if c_val is not None and int(c_val) in _MRP_ROUTE_TYPES:
                    result["routeType"] = _MRP_ROUTE_TYPES[int(c_val)]
            except (ValueError, TypeError, KeyError):
                pass

        if "routeType" not in result:
            result["routeType"] = DEFAULT_ROUTE_TYPE
    else:
        result["type"] = "showmap"

    return result


async def resolve_mapy_url(url: str) -> dict:
    if "/fnc/v1/" in url:
        return parse_mapy_url(url)
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = await client.get(url)
        resolved = str(response.url)
    if "/fnc/v1/" in resolved:
        return parse_mapy_url(resolved)
    # Try parsing as a legacy mapy.com URL
    result = _parse_legacy_mapy_url(resolved)

    # If a dim parameter is present, resolve the full shared item data
    dim_id = result.get("dim")
    if dim_id:
        dim_data = await resolve_dim(dim_id)
        if dim_data:
            # Merge dim data into the result (dim data takes priority)
            if dim_data.get("start"):
                result["start"] = dim_data["start"]
            if dim_data.get("end"):
                result["end"] = dim_data["end"]
            if dim_data.get("waypoints"):
                result["waypoints"] = dim_data["waypoints"]
            if dim_data.get("route_geometry"):
                result["route_geometry"] = dim_data["route_geometry"]
            if dim_data.get("totalLength"):
                result["totalLength"] = dim_data["totalLength"]
            if dim_data.get("totalTime"):
                result["totalTime"] = dim_data["totalTime"]
            if dim_data.get("routeType"):
                result["routeType"] = dim_data["routeType"]
            if dim_data.get("title"):
                result["dim_title"] = dim_data["title"]
            if dim_data.get("center") and "center" not in result:
                result["center"] = dim_data["center"]
            if dim_data.get("bbox"):
                result["bbox"] = dim_data["bbox"]
            # Mark as route if we have start+end
            if result.get("start") and result.get("end"):
                result["type"] = "route"

    return result


def tile_layer_config(mapset: str = DEFAULT_MAPSET) -> dict:
    mapset = normalize_mapset(mapset)
    url = DEFAULT_TILE_URL_TEMPLATE.format(mapset=mapset, z="{z}", x="{x}", y="{y}")
    api_key = os.getenv(API_KEY_ENV)
    if api_key and "apikey=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}apikey={api_key}"
    return {
        "url": url,
        "attribution": "&copy; <a href=\"https://www.seznam.cz\" target=\"_blank\">Seznam.cz, a.s.</a> &copy; <a href=\"https://www.openstreetmap.org/copyright\" target=\"_blank\">OpenStreetMap</a>",
    }


def build_route_shape(coordinates: Sequence[Sequence[float]], color: str = "#2f855a") -> str:
    path = ";".join([f"{lon},{lat}" for lon, lat in coordinates])
    return f"color:{color};width:3;path:[({path})]"


def normalize_mapset(mapset: Optional[str]) -> str:
    if not mapset:
        return DEFAULT_MAPSET
    normalized = mapset.strip().lower()
    return MAPSET_ALIASES.get(normalized, DEFAULT_MAPSET)
