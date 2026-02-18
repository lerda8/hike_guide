from typing import Optional, List
from datetime import datetime
from sqlmodel import Field, SQLModel, Relationship

class Tag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

    hike_tags: List["HikeTag"] = Relationship(back_populates="tag")

class HikeTag(SQLModel, table=True):
    hike_id: int = Field(foreign_key="hike.id", primary_key=True)
    tag_id: int = Field(foreign_key="tag.id", primary_key=True)

    hike: "Hike" = Relationship(back_populates="hike_tags")
    tag: Tag = Relationship(back_populates="hike_tags")

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    password_hash: str

    hikes: List["Hike"] = Relationship(back_populates="creator")
    ratings: List["Rating"] = Relationship(back_populates="user")
    photos: List["Photo"] = Relationship(back_populates="user")

class Hike(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: Optional[str] = None
    distance: float = 0.0 # in km
    elevation_gain: int = 0 # in meters
    difficulty: str = "Medium" # Easy, Medium, Hard
    start_name: Optional[str] = None
    end_name: Optional[str] = None
    start_lon: Optional[float] = None
    start_lat: Optional[float] = None
    end_lon: Optional[float] = None
    end_lat: Optional[float] = None
    mapy_url: Optional[str] = None
    route_type: str = "foot_hiking"
    mapset: str = "outdoor"
    country: Optional[str] = None # e.g. "Czech Republic", "Poland"
    created_by: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    creator: User = Relationship(back_populates="hikes")
    routes: List["Route"] = Relationship(back_populates="hike")
    photos: List["Photo"] = Relationship(back_populates="hike")
    ratings: List["Rating"] = Relationship(back_populates="hike")
    hike_tags: List["HikeTag"] = Relationship(back_populates="hike")

class Route(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hike_id: int = Field(foreign_key="hike.id")
    geojson_data: str # Stored as JSON string

    hike: Hike = Relationship(back_populates="routes")

class Photo(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hike_id: int = Field(foreign_key="hike.id")
    user_id: int = Field(foreign_key="user.id")
    filename: str
    caption: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    hike: Hike = Relationship(back_populates="photos")
    user: User = Relationship(back_populates="photos")

class Rating(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hike_id: int = Field(foreign_key="hike.id")
    user_id: int = Field(foreign_key="user.id")
    rating: int = Field(ge=1, le=5)
    nickname: str = Field(default="Anonymous")
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    hike: Hike = Relationship(back_populates="ratings")
    user: User = Relationship(back_populates="ratings")
