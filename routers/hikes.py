from fastapi import APIRouter, Depends, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select
from database import get_session
from models import Hike, User, Photo, Route, Rating
import shutil
import os
import uuid

router = APIRouter(prefix="/hikes", tags=["hikes"])
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def list_hikes(request: Request, session: Session = Depends(get_session)):
    hikes = session.exec(select(Hike)).all()
    return templates.TemplateResponse("hikes/list.html", {"request": request, "hikes": hikes})

@router.get("/new", response_class=HTMLResponse)
async def new_hike_form(request: Request):
    return templates.TemplateResponse("hikes/form.html", {"request": request})

@router.post("/new")
async def create_hike(
    request: Request,
    title: str = Form(...),
    description: str = Form(None),
    distance: float = Form(0.0),
    elevation_gain: int = Form(0),
    difficulty: str = Form("Medium"),
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

    hike = Hike(
        title=title,
        description=description,
        distance=distance,
        elevation_gain=elevation_gain,
        difficulty=difficulty,
        created_by=user.id
    )
    session.add(hike)
    session.commit()
    session.refresh(hike)

    if gpx_file and gpx_file.filename.endswith(".gpx"):
        try:
            import gpxpy
            import json
            
            content = await gpx_file.read()
            gpx = gpxpy.parse(content)
            
            # Extract tracks/segments to GeoJSON LineString
            coordinates = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        # GeoJSON expects [lon, lat, ele]
                        coordinates.append([point.longitude, point.latitude])
            
            if coordinates:
                geojson = {
                    "type": "LineString",
                    "coordinates": coordinates
                }
                
                from models import Route
                route = Route(hike_id=hike.id, geojson_data=json.dumps(geojson))
                session.add(route)
                session.commit()
        except Exception as e:
            print(f"Error parsing GPX: {e}")
            # TODO: Flash error message
            pass

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
    
    return templates.TemplateResponse("hikes/detail.html", {"request": request, "hike": hike, "route_json": route_json})
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
    
    if photo_file:
        # Generate unique filename
        ext = os.path.splitext(photo_file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join("static", "uploads", filename)
        
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
    comment: str = Form(None),
    session: Session = Depends(get_session)
):
    hike = session.get(Hike, hike_id)
    if not hike:
        return HTMLResponse("Hike not found", status_code=404)

    # TODO: Auth check. For now, check if user already rated? 
    # Or just let them spam ratings since it's a demo.
    user = session.exec(select(User).where(User.username == "demo")).first()
    
    new_rating = Rating(
        hike_id=hike.id,
        user_id=user.id,
        rating=rating,
        comment=comment
    )
    session.add(new_rating)
    session.commit()
        
    return RedirectResponse(url=f"/hikes/{hike.id}", status_code=303)
