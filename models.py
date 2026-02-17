from typing import Optional, List
from datetime import datetime
from sqlmodel import Field, SQLModel, Relationship

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
    status: str = "Planned" # Planned, Completed
    created_by: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    creator: User = Relationship(back_populates="hikes")
    routes: List["Route"] = Relationship(back_populates="hike")
    photos: List["Photo"] = Relationship(back_populates="hike")
    ratings: List["Rating"] = Relationship(back_populates="hike")

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
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    hike: Hike = Relationship(back_populates="ratings")
    user: User = Relationship(back_populates="ratings")
