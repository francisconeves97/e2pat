# Emerging Patterns App

## Installation:

Python 3+

All dependencies defined in **requirements.txt**. You can install them by:

```
$ conda create --name <env> --file requirements.txt
$ conda activate <env>
```

## Usage:

After installing the required dependencies and activating your freshly created environment you should be able to run our app. 

You can see our interface for querying road traffic data by running and accessing http://127.0.0.1:8051/:

```
$ python emerging_patterns.py
```

You can test our solution by using the example data available at the `data/` folder. First you should run the interface and access http://127.0.0.1:8050/:

```
$ python emerging_patterns_from_csv.py
```

After accessing the interface choose to upload a file, then navigate to `data/` and choose `example-dataset.csv`.
