import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RackLocation(Base):
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    location = Column(Geometry('POINT'), nullable=True)
    numracks = Column(Integer, default=0)
    __tablename__ = 'rack_locations'
