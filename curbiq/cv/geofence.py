"""No-parking-zone geofencing + pixel->GPS homography for camera ingestion."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon


@dataclass
class NoParkingZone:
    """A no-parking polygon in image pixel coordinates."""
    zone_id: str
    polygon_xy: list[tuple[float, float]]          # [(x, y), ...] image px
    offence_codes: list[int] = field(default_factory=lambda: [113])  # default NO PARKING

    def __post_init__(self):
        self._poly = Polygon(self.polygon_xy)

    def contains_point(self, x: float, y: float) -> bool:
        return self._poly.covers(Point(x, y))


class Homography:
    """Map image pixels -> world (lon, lat) from >=4 point correspondences (DLT)."""

    def __init__(self, src_px: list[tuple[float, float]],
                 dst_lonlat: list[tuple[float, float]]):
        if len(src_px) < 4 or len(dst_lonlat) < 4:
            raise ValueError("need >= 4 point correspondences")
        A = []
        for (x, y), (X, Y) in zip(src_px, dst_lonlat):
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
        _, _, Vt = np.linalg.svd(np.asarray(A, float))
        H = Vt[-1].reshape(3, 3)
        self.H = H / H[2, 2]

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        v = self.H @ np.array([x, y, 1.0])
        v /= v[2]
        return float(v[0]), float(v[1])


@dataclass
class CameraConfig:
    """A fixed enforcement camera: where it is and what it watches."""
    camera_id: str
    location: str                                  # human-readable address
    police_station: str
    junction_name: str | None = None
    center_code: str | None = None
    lat: float | None = None                       # used when no homography
    lon: float | None = None
    zones: list[NoParkingZone] = field(default_factory=list)
    homography: Homography | None = None

    def world_of(self, x: float, y: float) -> tuple[float, float]:
        """Map a ground pixel to (lat, lon): homography if set, else the fixed camera point."""
        if self.homography is not None:
            lon, lat = self.homography.to_lonlat(x, y)
            return lat, lon
        return self.lat, self.lon

    def zone_for(self, x: float, y: float) -> NoParkingZone | None:
        for z in self.zones:
            if z.contains_point(x, y):
                return z
        return None
