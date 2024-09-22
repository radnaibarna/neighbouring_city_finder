import pandas as pd
from rtree import index as rtree_idx
from math import radians, cos, sin, sqrt, atan2

EARTH_RADIUS = 6371.0


class Rtree:
    def __init__(self, cities_dataframe: pd.DataFrame) -> None:
        self._cities_df: pd.DataFrame = cities_dataframe
        self._rtree = rtree_idx.Index()

        self._init_rtree()
        
    def _init_rtree(self) -> None:
        for index, rows in self._cities_df.iterrows():
            self._rtree.insert(int(rows["id"]), (rows["lat"], rows["lng"], rows["lat"], rows["lng"]))

    @staticmethod
    def _create_bounding_box_for_coordinates(lat: float, lng: float, radius_km: float):
        lat_change = (radius_km / EARTH_RADIUS) * (180 / 3.14159)
        lng_change = lat_change / cos(radians(lat))
        return(lat - lat_change, lng - lng_change, lat + lat_change, lng + lng_change)
    
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = EARTH_RADIUS * c
        return distance
    
    def _get_candidates(self, bounding_box: tuple[float,float,float,float]):
        return list(self._rtree.intersection(bounding_box))
    
    def _filter_candidates(self, candidates: list[int], city_lat: float, city_lng: float, radius: float):
        close_city_ids = []
        for candidate in candidates:
            candidate_row = self._cities_df.loc[self._cities_df["id"] == candidate]
            candidate_lat = candidate_row["lat"]
            candidate_lng = candidate_row ["lng"]
            distance = self.haversine(lat1=city_lat, lon1=city_lng, lat2=candidate_lat, lon2=candidate_lng)
            if distance <= radius:
                close_city_ids.append(candidate)

        return close_city_ids

    def get_close_cities(self, city_id: int, radius: float):
        city_row = self._cities_df.loc[self._cities_df["id"] == city_id]
        city_lat: float = float(city_row["lat"])
        city_lng: float = float(city_row["lng"])

        bounding_box = self._create_bounding_box_for_coordinates(
            lat=city_lat,
            lng=city_lng,
            radius_km=radius
        )
        potential_candidates = self._get_candidates(bounding_box=bounding_box)
        filtered_candidates = self._filter_candidates(
            candidates=potential_candidates,
            city_lat=city_lat,
            city_lng=city_lng,
            radius=radius
        )

        return filtered_candidates
    


