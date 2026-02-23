from typing import Protocol, List
from sqlalchemy.orm import Session
from .models import PlantTimeseries


class TimeseriesDataSource(Protocol):
  def latest_point(self, session: Session, plant_id: int) -> PlantTimeseries | None:
    ...

  def history(
    self,
    session: Session,
    plant_id: int,
    limit: int = 288,
  ) -> List[PlantTimeseries]:
    ...


class DbTimeseriesDataSource:
  def latest_point(self, session: Session, plant_id: int) -> PlantTimeseries | None:
    q = (
      session.query(PlantTimeseries)
      .filter(PlantTimeseries.plant_id == plant_id)
      .order_by(PlantTimeseries.timestamp.desc())
    )
    return q.first()

  def history(
    self,
    session: Session,
    plant_id: int,
    limit: int = 288,
  ) -> List[PlantTimeseries]:
    q = (
      session.query(PlantTimeseries)
      .filter(PlantTimeseries.plant_id == plant_id)
      .order_by(PlantTimeseries.timestamp.desc())
      .limit(limit)
    )
    rows = q.all()
    rows.reverse()
    return rows

