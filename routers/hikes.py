from fastapi import APIRouter, Depends, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select, func
from database import get_session
from models import Hike, User, Photo, Route, Rating, Tag, HikeTag
import shutil
import os
import uuid
import json

from services import mapy_cz

router = APIRouter(prefix="/hikes", tags=["hikes"])
templates = Jinja2Templates(directory="templates")


def _extract_coordinates(route_geojson: dict) -> list[list[float]]:
    if not route_geojson:
        return []
    if route_geojson.get("type") == "Feature":
        geometry = route_geojson.get("geometry", {})
    else:
        geometry = route_geojson
    if geometry.get("type") == "LineString":
        return geometry.get("coordinates", [])
    return []


def _sample_waypoints(coordinates: list[list[float]], max_points: int = 15) -> list[list[float]]:
    if len(coordinates) <= max_points:
        return coordinates
    step = max(1, len(coordinates) // max_points)
    sampled = [coordinates[idx] for idx in range(0, len(coordinates), step)]
    return sampled[:max_points]


def _estimate_hiking_time(distance_km: float, elevation_gain_m: int) -> str | None:
    """Naismith's rule: 4 km/h + 1 h per 600 m ascent."""
    if not distance_km and not elevation_gain_m:
        return None
    hours = (distance_km or 0) / 4.0 + (elevation_gain_m or 0) / 600.0
    if hours < 1:
        return f"{int(round(hours * 60))} min"
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 0:
        return f"{h} h"
    return f"{h} h {m} min"

@router.get("/", response_class=HTMLResponse)
async def list_hikes(
    request: Request,
    session: Session = Depends(get_session),
    # Filters
    status: str = Query("all"),  # all, planned, completed
    difficulty: str = Query("all"),  # all, easy, medium, hard
    min_distance: float = Query(0),
    max_distance: float = Query(1000),
    min_elevation: int = Query(0),
    max_elevation: int = Query(10000),
    min_rating: float = Query(0),
    country: str = Query(""),
    tags: str = Query(""),  # comma-separated tag names
    sort: str = Query("newest"),  # newest, distance, elevation, difficulty, rating
):
    query = select(Hike)

    # Apply filters
    if status == "planned":
        query = query.where(Hike.status == "Planned")
    elif status == "completed":
        query = query.where(Hike.status == "Completed")

    if difficulty != "all":
        query = query.where(Hike.difficulty == difficulty.capitalize())

    query = query.where(Hike.distance >= min_distance).where(Hike.distance <= max_distance)
    query = query.where(Hike.elevation_gain >= min_elevation).where(Hike.elevation_gain <= max_elevation)

    if country:
        query = query.where(Hike.country == country)

    # Apply rating filter (requires join with ratings)
    if min_rating > 0:
        # Subquery: hikes with avg rating >= min_rating
        rating_subq = select(Hike.id).join(Rating).group_by(Hike.id).having(func.avg(Rating.rating) >= min_rating)
        query = query.where(Hike.id.in_(rating_subq))

    # Apply tag filter
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        for tag_name in tag_list:
            tag = session.exec(select(Tag).where(Tag.name == tag_name)).first()
            if tag:
                tag_hike_ids = session.exec(select(HikeTag.hike_id).where(HikeTag.tag_id == tag.id)).all()
                query = query.where(Hike.id.in_(tag_hike_ids))

    # Apply sorting
    if sort == "distance":
        query = query.order_by(Hike.distance)
    elif sort == "elevation":
        query = query.order_by(Hike.elevation_gain.desc())
    elif sort == "difficulty":
        # Easy < Medium < Hard
        difficulty_order = {"Easy": 1, "Medium": 2, "Hard": 3}
        query = query.order_by(Hike.difficulty)
    elif sort == "rating":
        rating_subq = select(Hike.id, func.avg(Rating.rating).label("avg_rating")).join(Rating, isouter=True).group_by(Hike.id)
        # This is complex; for now just sort by newest
        query = query.order_by(Hike.created_at.desc())
    else:  # newest
        query = query.order_by(Hike.created_at.desc())

    hikes = session.exec(query).all()

    # Get all countries and tags for filter UI
    all_countries = session.exec(select(Hike.country).distinct()).all()
    all_countries = sorted([c for c in all_countries if c])
    all_tags = session.exec(select(Tag)).all()

    return templates.TemplateResponse("hikes/list.html", {
        "request": request,
        "hikes": hikes,
        "all_countries": all_countries,
        "all_tags": all_tags,
        "filters": {
            "status": status,
            "difficulty": difficulty,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "min_elevation": min_elevation,
            "max_elevation": max_elevation,
            "min_rating": min_rating,
            "country": country,
            "tags": tags,
            "sort": sort,
        },
    })


@router.get("/map", response_class=HTMLResponse)
async def all_hikes_map(request: Request, session: Session = Depends(get_session)):
    """Show all hikes on a single map."""
    hikes = session.exec(select(Hike)).all()
    # Build lightweight data for the map (id, title, start/end coords, route geometry)
    hike_markers = []
    for h in hikes:
        entry = {
            "id": h.id,
            "title": h.title,
            "distance": h.distance,
            "elevation_gain": h.elevation_gain,
            "difficulty": h.difficulty,
            "route_type": h.route_type,
            "estimated_time": _estimate_hiking_time(h.distance, h.elevation_gain),
            "start_name": h.start_name,
            "end_name": h.end_name,
        }
        if h.start_lon is not None and h.start_lat is not None:
            entry["start"] = [h.start_lat, h.start_lon]
        if h.end_lon is not None and h.end_lat is not None:
            entry["end"] = [h.end_lat, h.end_lon]
        # Get route geometry
        route = session.exec(select(Route).where(Route.hike_id == h.id)).first()
        if route:
            try:
                geojson = json.loads(route.geojson_data)
                entry["route"] = geojson
            except Exception:
                pass
        hike_markers.append(entry)

    tile_config = mapy_cz.tile_layer_config("outdoor")
    return templates.TemplateResponse("hikes/map.html", {
        "request": request,
        "hike_markers": hike_markers,
        "tile_url": tile_config["url"],
        "tile_attribution": tile_config["attribution"],
    })


@router.get("/mapy/suggest")
async def mapy_suggest(query: str = Query("", min_length=1)):
    suggestions = await mapy_cz.suggest(query, limit=7)
    return JSONResponse(suggestions)


@router.get("/mapy/resolve-url")
async def mapy_resolve_url(url: str = Query(...), session: Session = Depends(get_session)):
    """Resolve a mapy.com URL and return parsed start/end/routeType/mapset."""
    try:
        resolved = await mapy_cz.resolve_mapy_url(url)

        # ── Duplicate detection ────────────────────────────────────────
        # Check if any existing hike already uses this URL (or the resolved long URL)
        urls_to_check = [url]
        if resolved.get("resolved_url") and resolved["resolved_url"] != url:
            urls_to_check.append(resolved["resolved_url"])
        for check_url in urls_to_check:
            existing = session.exec(
                select(Hike).where(Hike.mapy_url == check_url)
            ).first()
            if existing:
                resolved["duplicate"] = {
                    "id": existing.id,
                    "title": existing.title,
                }
                break
        # Also reverse-geocode start and end to get names
        if resolved.get("start"):
            try:
                rg = await mapy_cz.reverse_geocode(resolved["start"][0], resolved["start"][1])
                if rg.get("items"):
                    resolved["start_name"] = rg["items"][0].get("name", "")
            except Exception:
                pass
        if resolved.get("end"):
            try:
                rg = await mapy_cz.reverse_geocode(resolved["end"][0], resolved["end"][1])
                if rg.get("items"):
                    resolved["end_name"] = rg["items"][0].get("name", "")
            except Exception:
                pass
        # Reverse-geocode center for legacy URLs that only have center
        if resolved.get("center") and not resolved.get("start"):
            try:
                rg = await mapy_cz.reverse_geocode(resolved["center"][0], resolved["center"][1])
                if rg.get("items"):
                    resolved["center_name"] = rg["items"][0].get("name", "")
            except Exception:
                pass
        # For dim-resolved routes: use dim_title if available
        if resolved.get("dim_title") and not resolved.get("start_name"):
            resolved["dim_title"] = resolved["dim_title"]
        # Count route points for UI feedback
        if resolved.get("route_geometry"):
            resolved["route_points_count"] = len(resolved["route_geometry"])
            # Don't send full geometry in the preview response (too large)
            del resolved["route_geometry"]
        return JSONResponse(resolved)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@router.get("/mapy/static-map/{hike_id}")
async def mapy_static_map(hike_id: int, session: Session = Depends(get_session)):
    hike = session.get(Hike, hike_id)
    if not hike:
        return Response(status_code=404)

    route = session.exec(select(Route).where(Route.hike_id == hike_id)).first()
    coordinates = []
    if route:
        coordinates = _extract_coordinates(json.loads(route.geojson_data))

    markers = []
    if hike.start_lon is not None and hike.start_lat is not None:
        markers.append(f"color:green;label:S;{hike.start_lon},{hike.start_lat}")
    if hike.end_lon is not None and hike.end_lat is not None:
        markers.append(f"color:red;label:E;{hike.end_lon},{hike.end_lat}")

    shapes = []
    if coordinates:
        shapes.append(mapy_cz.build_route_shape(coordinates))

    # Compute bounds from ALL route coordinates so the full trail is visible
    bounds = None
    if coordinates:
        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]
        bounds = [
            [min(lons), min(lats)],
            [max(lons), max(lats)],
        ]
    elif hike.start_lon is not None and hike.start_lat is not None and hike.end_lon is not None and hike.end_lat is not None:
        # Both start and end: use them as bounds
        bounds = [
            [min(hike.start_lon, hike.end_lon), min(hike.start_lat, hike.end_lat)],
            [max(hike.start_lon, hike.end_lon), max(hike.start_lat, hike.end_lat)],
        ]
    elif hike.start_lon is not None and hike.start_lat is not None:
        # Only start: zoom around it with a small padding
        padding = 0.01
        bounds = [
            [hike.start_lon - padding, hike.start_lat - padding],
            [hike.start_lon + padding, hike.start_lat + padding],
        ]

    # Can't generate a static map without at least one coordinate
    if not bounds and not coordinates:
        return Response(status_code=404)

    mapset = mapy_cz.normalize_mapset(hike.mapset or "outdoor")
    try:
        content, content_type = await mapy_cz.get_static_map_image(
            lon_lat_bounds=bounds,
            markers=markers,
            shapes=shapes,
            width=640,
            height=360,
            mapset=mapset,
        )
        return Response(content=content, media_type=content_type)
    except Exception:
        return Response(status_code=404)

@router.get("/new", response_class=HTMLResponse)
async def new_hike_form(request: Request):
    return templates.TemplateResponse("hikes/form.html", {"request": request})

@router.post("/new")
async def create_hike(
    request: Request,
    title: str = Form(...),
    description: str = Form(None),
    start_location: str = Form(None),
    end_location: str = Form(None),
    start_lon: float = Form(None),
    start_lat: float = Form(None),
    end_lon: float = Form(None),
    end_lat: float = Form(None),
    route_type: str = Form("foot_hiking"),
    mapset: str = Form("outdoor"),
    country: str = Form(None),
    tags: str = Form(None),
    mapy_url: str = Form(None),
    gpx_file: UploadFile = File(None),
    session: Session = Depends(get_session)
):
    # TODO: Get actual user from auth
    user = session.exec(select(User).where(User.username == "demo")).first()
    if not user:
        # Fallback if demo user deleted
        user = User(username="demo", email="demo@example.com", password_hash="dummy")
        session.add(user)
        session.commit()
        session.refresh(user)

    # ── Resolve Mapy URL ───────────────────────────────────────────────
    resolved_mapy = None
    plan_route = False

    if mapy_url:
        try:
            resolved_mapy = await mapy_cz.resolve_mapy_url(mapy_url)
        except Exception:
            resolved_mapy = {"raw": mapy_url}

    if resolved_mapy and resolved_mapy.get("start"):
        start_lon, start_lat = resolved_mapy["start"]
    if resolved_mapy and resolved_mapy.get("end"):
        end_lon, end_lat = resolved_mapy["end"]
    if resolved_mapy and resolved_mapy.get("center") and start_lon is None:
        start_lon, start_lat = resolved_mapy["center"]
    if resolved_mapy and resolved_mapy.get("mapset"):
        mapset = resolved_mapy["mapset"]
    if resolved_mapy and resolved_mapy.get("routeType"):
        route_type = resolved_mapy["routeType"]

    # Use pre-resolved route geometry from dim (shared items) if available
    dim_route_geometry = None
    dim_route_length = None
    if resolved_mapy and resolved_mapy.get("route_geometry"):
        dim_route_geometry = resolved_mapy["route_geometry"]
    if resolved_mapy and resolved_mapy.get("totalLength"):
        dim_route_length = resolved_mapy["totalLength"]

    # Always plan route when we have BOTH start and end from URL
    if resolved_mapy:
        has_start = resolved_mapy.get("start") is not None
        has_end = resolved_mapy.get("end") is not None
        if has_start and has_end:
            plan_route = True

    # ── Reverse-geocode location names ─────────────────────────────────
    if start_lon is not None and start_lat is not None and not start_location:
        try:
            rg = await mapy_cz.reverse_geocode(start_lon, start_lat)
            if rg.get("items"):
                start_location = rg["items"][0].get("name", "Start")
        except Exception:
            start_location = "Start"

    if end_lon is not None and end_lat is not None and not end_location:
        try:
            rg = await mapy_cz.reverse_geocode(end_lon, end_lat)
            if rg.get("items"):
                end_location = rg["items"][0].get("name", "End")
        except Exception:
            end_location = "End"

    mapset = mapy_cz.normalize_mapset(mapset)
    hike = Hike(
        title=title,
        description=description,
        distance=0.0,
        elevation_gain=0,
        difficulty="Medium",
        start_name=start_location,
        end_name=end_location,
        start_lon=start_lon,
        start_lat=start_lat,
        end_lon=end_lon,
        end_lat=end_lat,
        mapy_url=mapy_url,
        route_type=route_type,
        mapset=mapset,
        country=country if country else None,
        created_by=user.id,
    )
    session.add(hike)
    session.commit()
    session.refresh(hike)

    # ── Add tags ───────────────────────────────────────────────────────────
    if tags:
        tag_names = [t.strip() for t in tags.split(",") if t.strip()]
        for tag_name in tag_names:
            # Get or create tag
            tag = session.exec(select(Tag).where(Tag.name == tag_name)).first()
            if not tag:
                tag = Tag(name=tag_name)
                session.add(tag)
                session.commit()
                session.refresh(tag)
            # Link tag to hike
            hike_tag = HikeTag(hike_id=hike.id, tag_id=tag.id)
            session.add(hike_tag)
        session.commit()

    route_geojson = None
    route_length_m = None

    # ── GPX handling (always snap to trails) ───────────────────────────
    if gpx_file and gpx_file.filename and gpx_file.filename.endswith(".gpx"):
        try:
            import gpxpy
            
            content = await gpx_file.read()
            gpx = gpxpy.parse(content)
            
            coordinates = []
            # Parse tracks
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        coordinates.append([point.longitude, point.latitude])
            # Fallback: parse routes if no tracks
            if not coordinates:
                for route_elem in gpx.routes:
                    for point in route_elem.points:
                        coordinates.append([point.longitude, point.latitude])
            # Fallback: parse waypoints if nothing else
            if not coordinates and gpx.waypoints:
                for point in gpx.waypoints:
                    coordinates.append([point.longitude, point.latitude])

            # Use gpxpy's built-in length as baseline distance
            gpx_length_m = gpx.length_3d() or gpx.length_2d()
            if gpx_length_m and gpx_length_m > 0:
                route_length_m = gpx_length_m
            
            if coordinates:
                if start_lon is None or start_lat is None:
                    start_lon, start_lat = coordinates[0]
                    hike.start_lon = start_lon
                    hike.start_lat = start_lat
                if end_lon is None or end_lat is None:
                    end_lon, end_lat = coordinates[-1]
                    hike.end_lon = end_lon
                    hike.end_lat = end_lat

                # Auto reverse-geocode GPX endpoints
                if not hike.start_name:
                    try:
                        rg = await mapy_cz.reverse_geocode(start_lon, start_lat)
                        if rg.get("items"):
                            hike.start_name = rg["items"][0].get("name", "Start")
                    except Exception:
                        hike.start_name = "Start"
                if not hike.end_name:
                    try:
                        rg = await mapy_cz.reverse_geocode(end_lon, end_lat)
                        if rg.get("items"):
                            hike.end_name = rg["items"][0].get("name", "End")
                    except Exception:
                        hike.end_name = "End"

                session.add(hike)
                session.commit()

                # Always snap GPX to trails
                waypoints = _sample_waypoints(coordinates, max_points=15)
                try:
                    route_result = await mapy_cz.plan_route(
                        start=[start_lon, start_lat],
                        end=[end_lon, end_lat],
                        waypoints=waypoints[1:-1] if len(waypoints) > 2 else None,
                        route_type=route_type,
                    )
                    route_geojson = route_result.geometry
                    route_length_m = route_result.length_meters
                except Exception as exc:
                    print(f"Mapy.cz snap-to-trail failed, using raw GPX: {exc}")
                    route_geojson = {"type": "LineString", "coordinates": coordinates}
        except Exception as e:
            print(f"Error parsing GPX: {e}")

    # ── Plan route from coordinates ────────────────────────────────────
    # Use pre-resolved dim geometry if available (from mapy.com shared items)
    if route_geojson is None and dim_route_geometry:
        route_geojson = {"type": "LineString", "coordinates": dim_route_geometry}
        route_length_m = dim_route_length

    if route_geojson is None and plan_route and start_lon is not None and start_lat is not None and end_lon is not None and end_lat is not None:
        try:
            route_result = await mapy_cz.plan_route(
                start=[start_lon, start_lat],
                end=[end_lon, end_lat],
                route_type=route_type,
            )
            route_geojson = route_result.geometry
            route_length_m = route_result.length_meters
        except Exception as exc:
            print(f"Mapy.cz routing failed: {exc}")

    # ── Save route ─────────────────────────────────────────────────────
    if route_geojson:
        route = Route(hike_id=hike.id, geojson_data=json.dumps(route_geojson))
        session.add(route)
        session.commit()

    # ── Auto-compute distance ──────────────────────────────────────────
    if route_length_m:
        hike.distance = round(route_length_m / 1000.0, 2)
    elif route_geojson:
        # Fallback: estimate from coordinates using haversine
        coords = _extract_coordinates(route_geojson)
        if not coords and isinstance(route_geojson, dict) and route_geojson.get("type") == "LineString":
            coords = route_geojson.get("coordinates", [])
        if coords:
            import math
            total = 0.0
            for i in range(1, len(coords)):
                lon1, lat1 = math.radians(coords[i-1][0]), math.radians(coords[i-1][1])
                lon2, lat2 = math.radians(coords[i][0]), math.radians(coords[i][1])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                total += 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            hike.distance = round(total / 1000.0, 2)

    # ── Auto-compute elevation gain ────────────────────────────────────
    if route_geojson:
        try:
            coordinates = _extract_coordinates(route_geojson)
            if not coordinates and isinstance(route_geojson, dict) and route_geojson.get("type") == "LineString":
                coordinates = route_geojson.get("coordinates", [])
            if coordinates:
                elevation_profile = await mapy_cz.elevation_profile_for_route(coordinates)
                if elevation_profile and len(elevation_profile) > 1:
                    gain = 0
                    for i in range(1, len(elevation_profile)):
                        diff = elevation_profile[i]["elevation"] - elevation_profile[i-1]["elevation"]
                        if diff > 0:
                            gain += diff
                    hike.elevation_gain = int(round(gain))
        except Exception as exc:
            print(f"Elevation computation failed: {exc}")
            # Fallback: use gpxpy's uphill calculation if GPX had elevation data
            if 'gpx' in dir() and gpx:
                try:
                    uphill, _, _ = gpx.get_uphill_downhill()
                    if uphill and uphill > 0:
                        hike.elevation_gain = int(round(uphill))
                except Exception:
                    pass

    # ── Auto-classify difficulty ───────────────────────────────────────
    dist_km = hike.distance or 0
    elev_m = hike.elevation_gain or 0
    if dist_km > 20 or elev_m > 800:
        hike.difficulty = "Hard"
    elif dist_km > 8 or elev_m > 400:
        hike.difficulty = "Medium"
    else:
        hike.difficulty = "Easy"

    session.add(hike)
    session.commit()

    return RedirectResponse(url=f"/hikes/{hike.id}", status_code=303)

@router.get("/{hike_id}", response_class=HTMLResponse)
async def hike_detail(request: Request, hike_id: int, session: Session = Depends(get_session)):
    hike = session.get(Hike, hike_id)
    if not hike:
        return HTMLResponse("Hike not found", status_code=404)
        
    # Get associated route
    from models import Route
    route = session.exec(select(Route).where(Route.hike_id == hike_id)).first()
    route_json = route.geojson_data if route else None
    route_geojson = None
    elevation_profile = []
    if route_json:
        try:
            route_geojson = json.loads(route_json)
            coordinates = _extract_coordinates(route_geojson)
            elevation_profile = await mapy_cz.elevation_profile_for_route(coordinates)
        except Exception as exc:
            print(f"Elevation profile failed: {exc}")

    mapset = mapy_cz.normalize_mapset(hike.mapset or "outdoor")
    tile_config = mapy_cz.tile_layer_config(mapset)

    mapy_show_url = None
    mapy_route_url = None
    if hike.start_lon is not None and hike.start_lat is not None:
        mapy_show_url = mapy_cz.build_mapy_url_showmap(
            [hike.start_lon, hike.start_lat],
            zoom=13,
            mapset=mapset,
        )
    if hike.start_lon is not None and hike.start_lat is not None and hike.end_lon is not None and hike.end_lat is not None:
        mapy_route_url = mapy_cz.build_mapy_url_route(
            [hike.start_lon, hike.start_lat],
            [hike.end_lon, hike.end_lat],
            route_type=hike.route_type or "foot_hiking",
            mapset=mapset,
        )

    # Compute average rating
    avg_rating = None
    rating_count = len(hike.ratings)
    if rating_count > 0:
        avg_rating = round(sum(r.rating for r in hike.ratings) / rating_count, 1)

    return templates.TemplateResponse(
        "hikes/detail.html",
        {
            "request": request,
            "hike": hike,
            "route_geojson": route_geojson,
            "elevation_profile": elevation_profile,
            "tile_url": tile_config["url"],
            "tile_attribution": tile_config["attribution"],
            "mapy_show_url": mapy_show_url,
            "mapy_route_url": mapy_route_url,
            "estimated_time": _estimate_hiking_time(hike.distance, hike.elevation_gain),
            "avg_rating": avg_rating,
            "rating_count": rating_count,
        },
    )
@router.post("/{hike_id}/photos")
async def upload_photo(
    hike_id: int,
    photo_file: UploadFile = File(...),
    caption: str = Form(None),
    session: Session = Depends(get_session)
):
    hike = session.get(Hike, hike_id)
    if not hike:
        return HTMLResponse("Hike not found", status_code=404)

    # TODO: Auth check
    user = session.exec(select(User).where(User.username == "demo")).first()
    
    if photo_file and photo_file.filename:
        # Validate file type
        allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if photo_file.content_type not in allowed_types:
            return RedirectResponse(url=f"/hikes/{hike.id}", status_code=303)

        # Generate unique filename
        ext = os.path.splitext(photo_file.filename)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            ext = ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        
        # Save file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(photo_file.file, buffer)
            
        # Create DB record
        photo = Photo(
            hike_id=hike.id,
            user_id=user.id,
            filename=filename,
            caption=caption
        )
        session.add(photo)
        session.commit()
        
    return RedirectResponse(url=f"/hikes/{hike.id}", status_code=303)

@router.post("/{hike_id}/ratings")
async def submit_rating(
    hike_id: int,
    rating: int = Form(...),
    nickname: str = Form(...),
    comment: str = Form(None),
    session: Session = Depends(get_session)
):
    hike = session.get(Hike, hike_id)
    if not hike:
        return HTMLResponse("Hike not found", status_code=404)

    user = session.exec(select(User).where(User.username == "demo")).first()

    new_rating = Rating(
        hike_id=hike.id,
        user_id=user.id,
        nickname=nickname.strip() or "Anonymous",
        rating=rating,
        comment=comment
    )
    session.add(new_rating)
    session.commit()
        
    return RedirectResponse(url=f"/hikes/{hike.id}", status_code=303)


@router.post("/{hike_id}/delete")
async def delete_hike(hike_id: int, session: Session = Depends(get_session)):
    hike = session.get(Hike, hike_id)
    if not hike:
        return HTMLResponse("Hike not found", status_code=404)

    # Delete associated photos (files + DB records)
    for photo in hike.photos:
        filepath = os.path.join("static", "uploads", photo.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        session.delete(photo)

    # Delete associated ratings
    for rating in hike.ratings:
        session.delete(rating)

    # Delete associated routes
    for route in hike.routes:
        session.delete(route)

    session.delete(hike)
    session.commit()

    return RedirectResponse(url="/hikes/", status_code=303)
