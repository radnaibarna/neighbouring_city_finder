# Closest Cities Finder

This Python script finds the cities in a given radius to a given geographical coordinate using the Rtree or KDTree algorithm. It takes a CSV file containing city data (name, latitude, longitude) as input, and generates a report CSV with details of each city's closest neighbors. 

Note: By default, it uses the Rtree object, which works with geospatially accurate numbers, and gives a more reliable report than the Kdtree algorithm. However, Kdtree algorithm is way faster than Rtree, so if speed is a factor, Kdtree is also usable

## Features
- Uses Rtree (Optionally KDTree) for fast spatial searching of nearest cities.
- Accepts a CSV file with city coordinates as input.
- Generates a CSV report showing the cities for each city in a given radius.
- Dockerized for easy deployment and execution without dependency management.
- Optionally a serialized object can be given as input, which represents the pre built kdtree or rtree object

## Requirements

- Python 3.11 (if not using Docker)
- The following Python packages (if not using Docker):
  - `structlog`
  - `scikit-learn`
  - `pandas`
  - `Rtree`
  - `numpy`

You can install the required libraries by running:

```bash
pip install structlog scikit-learn pandas Rtree numpy
```

### Running the Script

#### Using Python directly:

To run the script, execute the following command in your terminal or command prompt:

```bash
python closest_neighbour_finder.py --input_csv input.csv --output_csv output.csv --radius r --tree_obj_path rtree.pkl
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
    docker run -v /path/to/csv/uscities.csv:/app/input.csv -v /path/to/kdtree_obj/kdtree.pkl:/app/kdtree.pkl -v /home/golyvasiren/Public/mystuff/hw:/app closest_cities_finder --input_csv /app/input.csv --output_csv /app/output.csv --radius 50 --tree_obj_path /app/kdtree.pkl
    ```


