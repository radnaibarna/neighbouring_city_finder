# Closest Cities Finder

This Python script finds the closest `n` cities to a given geographical coordinate using the KDTree algorithm. It takes a CSV file containing city data (name, latitude, longitude) as input, and generates a report CSV with details of each city's closest neighbors. 

Note: since it is a pilot version, the generated report will only contain the ID-s of the closest neighbours of each city, if there is an ID provided in the input csv file

## Features
- Uses KDTree for fast spatial searching of nearest cities.
- Accepts a CSV file with city coordinates as input.
- Generates a CSV report showing the nearest `n` cities for each city.
- Dockerized for easy deployment and execution without dependency management.
- Optionally a serialized object can be given as input, which represents the pre built kdtree

## Requirements

- Python 3.11 (if not using Docker)
- The following Python packages (if not using Docker):
  - `structlog`
  - `scikit-learn`
  - `pandas`

You can install the required libraries by running:

```bash
pip install structlog scikit-learn pandas
```

### Running the Script

#### Using Python directly:

To run the script, execute the following command in your terminal or command prompt:

```bash
python closest_neighbour_finder.py --input_csv input.csv --output_csv output.csv --closest_n_neighbour n --kdtree_obj_path kdtree.pkl
```

#### Using Docker:

You can also run this script in a Docker container to avoid managing dependencies manually.

1. Build the Docker Image

    In the project directory (where the Dockerfile is located), build the Docker image:

    ```bash
    docker build -t closest-cities-finder .
    ```

2. Run the Docker Container

    After building the image, run the container with the following command:

    ```bash
    docker run -v /path/to/csv/uscities.csv:/app/input.csv -v /path/to/kdtree_obj/kdtree.pkl:/app/kdtree.pkl -v /home/golyvasiren/Public/mystuff/hw:/app closest_cities_finder --input_csv /app/input.csv --output_csv /app/output.csv --closest_n_neighbour 3 --kdtree_obj_path /app/kdtree.pkl
    ```


