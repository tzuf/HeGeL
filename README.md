# HeGeL: A Novel Dataset for Hebrew Geo-Location

## Data

The data can be found here - https://github.com/tzuf/HeGeL/tree/main/data/human.

The data contains three json files corresponding to three sets: train (Tel Aviv), dev (Haifa), and test (Jerusalem).

Each sample contains the following:

* content - place description.
* geometry - the wkt shape of the geolocation of the place.
* goal_point - the centroid of the geometry.


## Model

### Dependencies

* [Pytorch](https://pytorch.org/) - Machine learning library for Python-related dependencies
* [Anaconda](https://www.anaconda.com/download/) - Anaconda includes all the other Python-related dependencies
* [ArgParse](https://docs.python.org/3/library/argparse.html) - Command line parsing in Python

### Installation
Below are installation instructions under Anaconda.
IMPORTANT: We use python 3.8.15

 - Setup a fresh Anaconda environment and install packages: 
 ```sh
# create and switch to new anaconda env
$ conda create -n hegel python=3.8.15
$ source activate hegel

# install required packages
$ pip install -r requirements.txt
```

### Instructions
 - Here are the instructions to use the code base:
 
##### Train and Test Model:
 - To train the model with options, use the command line:
```sh
$ python train.py --options %(For the details of options)
$ python train.py [-h] [short_name_arg] %(For explanation on the commands)
```


