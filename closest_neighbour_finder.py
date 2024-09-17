from sklearn.neighbors import KDTree
import pandas as pd
import structlog
import argparse
import pickle


logger = structlog.get_logger()


class AbortMessage(Exception):
    pass


class DataWrapper:
    def __init__(self, used_columns: list[str], path_to_csv: str) -> None:
        self._used_columns = used_columns
        self._path_to_csv = path_to_csv
        self.coordinate_city_mapping: dict[str, dict] = {}
        self.coordinates_dataframe: pd.DataFrame = None

        self._read_data()

    def _read_data(self) -> bool:
        logger.info("Reading input csv...")
        try:
            required_dataframe = pd.read_csv(self._path_to_csv, usecols=self._used_columns)    
            self._create_coordinate_city_mapping(dataframe=required_dataframe)
        except (KeyError, ValueError) as exc:
            logger.error("Unexpected structure in the given csv file", error=exc)
            return False
        logger.info("Data initializing was successfull")
        
    def _create_coordinate_city_mapping(self, dataframe: pd.DataFrame) -> None:
        for index, row in dataframe.iterrows():
            hash_value = str(row["lat"]) + "_" + str(row["lng"])
            self.coordinate_city_mapping[hash_value] = {
                col_name: row[col_name] for col_name in self._used_columns
            }
        
        self.coordinates_dataframe = dataframe[["lat", "lng"]]

    def get_coordinates_by_index(self, index: int) -> tuple[float, float]:
        try:
            return float(self.coordinates_dataframe.iat[index, 0]), float(self.coordinates_dataframe.iat[index, 1])
            
        except IndexError as err:
            logger.error("Invalid index for the coordinates dataframe", error=err)

    def get_city_id_by_coordinates(self, lat: float, lng: float) -> str:
        hash_value = self._get_hash_from_coordinates(
            lat=lat,
            lng=lng
        )
        city_data = self.coordinate_city_mapping.get(hash_value)
        if not "id" in self._used_columns:
            logger.error(f"There was a faulty record: {city_data}")
            return None
        
        return int(city_data.get("id"))

    @staticmethod
    def _get_hash_from_coordinates(lat: float, lng: float):
        return str(lat) + "_" + str(lng)
    

class ClosestNeighbourCalculator:
    used_columns = ["id", "lat", "lng"]
 
    def __init__(self, path_to_input_csv: str, output_csv_name: str, kdtree_obj_path: str | None = None) -> None:
        self._input_csv: str = path_to_input_csv
        self._output_csv: str = output_csv_name
        self._kdtree_obj_path = kdtree_obj_path
        self._data_wrapper = DataWrapper(
            used_columns=self.used_columns,
            path_to_csv=self._input_csv
        )
        self._kdtree = None
        self._init_kdtree()
        

    def _init_kdtree(self):
        if self._kdtree_obj_path:
            logger.info("Using a serialized KDTree object")
            self._read_serialized_kdtree_object(kdtree_obj_path=self._kdtree_obj_path)
        else:
            logger.info("Creating KDTree object...")
            self._kdtree = KDTree(self._data_wrapper.coordinates_dataframe)
            self._create_serialized_object_from_kdtree()

    def _create_serialized_object_from_kdtree(self):
        logger.info("Starting serialization of kdtree object...")
        with open('kdtree.pkl', 'wb') as f:
            pickle.dump(self._kdtree, f)
        logger.info("Serialization of kdtree object was successfull. It can be found at kdtree.pkl")
        
        

    def _read_serialized_kdtree_object(self, kdtree_obj_path: str):
        logger.info("Starting deserialization of kdtree object...")
        try:
            with open("kdtree.pkl", "rb") as f:
                self._kdtree = pickle.load(f)
        except FileNotFoundError:
            logger.error("The given file for kdtree doesn't exists")
            raise AbortMessage
        except pickle.UnpicklingError as exc:
            logger.error("The given file for kdtree is corrupt or wrongly formatted", error=exc)
            raise AbortMessage
        except ValueError as exc:
            logger.error("Unsupported serialization format", error=exc)
            raise AbortMessage
        logger.info("Deserialization was successfull")

    def generate_report(self, closest_n_neighbour: int) -> None:
        output = []

        logger.info("Starting report genneration...")
        for city_data in self._data_wrapper.coordinates_dataframe.values:
            output_data = {
                "city_id": "",
                "closest_neighbours": []
            }
            distances, indexes_of_neighbours = self._kdtree.query([
                [city_data[0], city_data[1]]], 
                k=closest_n_neighbour+1
            )
            current_city_id = self._data_wrapper.get_city_id_by_coordinates(
                lat=city_data[0],
                lng=city_data[1]
            )

            output_data["city_id"] = current_city_id

            for city_index in indexes_of_neighbours[0]:
                lat, lng = self._data_wrapper.get_coordinates_by_index(index=city_index)
                city_id = self._data_wrapper.get_city_id_by_coordinates(
                    lat=lat,
                    lng=lng
                )
                if city_id != current_city_id:
                    output_data["closest_neighbours"].append(city_id)

            output.append(output_data)

        output_df = pd.DataFrame(output)
        output_df.to_csv(self._output_csv, index=False)
        logger.info(f"Report generation was successfull. The report is available at: {self._output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help='Input CSV file path')
    parser.add_argument('--output_csv', type=str, default='neighbour_report.csv', help='Output CSV file path')
    parser.add_argument('--closest_n_neighbour', type=int, default=3, help='The closest n city which should be included in the report')
    parser.add_argument('--kdtree_obj_path', type=str, default=None, help='Path to the serialized kdtree object')

    
    args = parser.parse_args()

    calculator = ClosestNeighbourCalculator(
        path_to_input_csv=args.input_csv,
        output_csv_name=args.output_csv,
        kdtree_obj_path=args.kdtree_obj_path
    )

    calculator.generate_report(closest_n_neighbour=args.closest_n_neighbour)



        
        
