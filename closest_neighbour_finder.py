from sklearn.neighbors import KDTree
import pandas as pd
import structlog
import argparse
import pickle
import numpy as np
from enum import Enum

from rtree_wrapper import Rtree

logger = structlog.get_logger()


class AbortMessage(Exception):
    pass


class AlgorithmEnum(Enum):
    KDTREE = "kdtree"
    RTREE = "rtree"


class DataWrapper:
    def __init__(self, used_columns: list[str], path_to_csv: str) -> None:
        self._used_columns = used_columns
        self._path_to_csv = path_to_csv
        self.coordinates_dataframe: pd.DataFrame = None

        self._read_data()

    def _read_data(self) -> bool:
        logger.info("Reading input csv...")
        try:
            self.coordinates_dataframe = pd.read_csv('uscities.csv', usecols=self._used_columns)

        except (KeyError, ValueError) as exc:
            logger.error("Unexpected structure in the given csv file", error=exc)
            return False
        logger.info("Data initializing was successfull")


class ClosestNeighbourCalculator:
    used_columns = ["id", "lat", "lng"]
 
    def __init__(self, path_to_input_csv: str, output_csv_name: str, tree_obj_path: str | None = None, algorithm: str = "rtree") -> None:
        self._input_csv: str = path_to_input_csv
        self._output_csv: str = output_csv_name
        self._tree_obj_path = tree_obj_path
        self._algorithm = algorithm
        self._data_wrapper = DataWrapper(
            used_columns=self.used_columns,
            path_to_csv=self._input_csv
        )        
        self._tree = None
        self._init_tree()


    def _init_tree(self):
        if self._algorithm == AlgorithmEnum.KDTREE.value:
            self._init_kdtree()
        elif self._algorithm == AlgorithmEnum.RTREE.value:
            self._init_rtree()
        
    def _init_rtree(self):
        if self._tree_obj_path:
            logger.info("Using a serialized RTree object")
            self._read_serialized_tree_object(tree_obj_path=self._tree_obj_path)
        else:
            logger.info("Creating RTree object...")
            self._tree = Rtree(cities_dataframe=self._data_wrapper.coordinates_dataframe)
            self._create_serialized_object_from_tree()
        logger.info("RTree object creation was successfull")  

    def _init_kdtree(self):
        if self._tree_obj_path:
            logger.info("Using a serialized KDTree object")
            self._read_serialized_tree_object(tree_obj_path=self._tree_obj_path)
        else:
            logger.info("Creating KDTree object...")
            self._tree = KDTree(self._data_wrapper.coordinates_dataframe[["lat", "lng"]])
            self._create_serialized_object_from_tree()

    def _create_serialized_object_from_tree(self):
        logger.info(f"Starting serialization of f{self._algorithm} object...")
        if self._algorithm == AlgorithmEnum.RTREE.value:
            with open(f'{self._algorithm}.pkl', 'wb') as f:
                pickle.dump(self._tree._rtree, f)
        else:
            with open(f'{self._algorithm}.pkl', 'wb') as f:
                pickle.dump(self._tree, f)
        logger.info("Serialization of kdtree object was successfull. It can be found at kdtree.pkl")

    def _read_serialized_tree_object(self, tree_obj_path: str):
        logger.info(f"Starting deserialization of {self._algorithm} object...")
        try:
            with open(tree_obj_path, "rb") as f:
                self._tree = pickle.load(f)
        except FileNotFoundError:
            logger.error("The given file doesn't exists")
            raise AbortMessage
        except pickle.UnpicklingError as exc:
            logger.error("The given file is corrupt or wrongly formatted", error=exc)
            raise AbortMessage
        except ValueError as exc:
            logger.error("Unsupported serialization format", error=exc)
            raise AbortMessage
        logger.info("Deserialization was successfull")

    def _rtree_calculation(self, radius: float) -> list[int]:
        output = []

        logger.info("Starting report generation...")
        for index, city in self._data_wrapper.coordinates_dataframe.iterrows():
            if index % 100 == 0:
                logger.info(f"Cities processed: {index}/{len(self._data_wrapper.coordinates_dataframe)}")
            output_data = {
                "city_id": "",
                "neighbours_in_radius": []
            }
            current_city_id = int(city["id"])
            close_cities = self._tree.get_close_cities(city_id=current_city_id, radius=radius)
            output_data["city_id"] = current_city_id
            output_data["neighbours_in_radius"] = close_cities
            output.append(output_data)

        return output

    def _kdtree_calculation(self, radius: float) -> list[int]:
        output = []

        logger.info("Starting report generation...")
        for index, city in self._data_wrapper.coordinates_dataframe.iterrows():
            if index % 100 == 0:
                logger.info(f"Cities processed: {index}/{len(self._data_wrapper.coordinates_dataframe)}")
            output_data = {
                "city_id": "",
                "neighbours_in_radius": []
            }
            indexes_of_neighbours = self._tree.query_radius(
                np.array([[float(city["lat"]), float(city["lng"])]]), 
                r=radius
            )
            current_city_id = int(city["id"])
            output_data["city_id"] = current_city_id
            output_data["neighbours_in_radius"] = indexes_of_neighbours[0].tolist()
            output.append(output_data)

        return output

    def generate_report(self, radius: float) -> None:
        if self._algorithm == AlgorithmEnum.KDTREE.value:
            output = self._kdtree_calculation(radius=radius)
        elif self._algorithm == AlgorithmEnum.RTREE.value:
            output = self._rtree_calculation(radius=radius)
        else:
            logger.error("Invalid algorithm")
            return
        output_df = pd.DataFrame(output)
        output_df.to_csv(self._output_csv, index=False)
        logger.info(f"Report generation was successfull. The report is available at: {self._output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help='Input CSV file path')
    parser.add_argument('--output_csv', type=str, default='neighbour_report.csv', help='Output CSV file path')
    parser.add_argument('--closest_n_neighbour', type=int, default=3, help='The closest n city which should be included in the report')
    parser.add_argument('--tree_obj_path', type=str, default=None, help='Path to the serialized kdtree object')
    parser.add_argument('--algorithm', type=str, default="rtree", help='kdtree or rtree')
    parser.add_argument('--radius', type=float, default=50.0, help='search radius')


    
    args = parser.parse_args()

    calculator = ClosestNeighbourCalculator(
        path_to_input_csv=args.input_csv,
        output_csv_name=args.output_csv,
        tree_obj_path=args.tree_obj_path,
        algorithm=args.algorithm
    )

    calculator.generate_report(radius=args.radius)



        
        
