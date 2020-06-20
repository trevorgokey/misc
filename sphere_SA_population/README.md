
# Description

A quick utility to calculate the surface area of a sphere covered by a collection of points.

Originally written to estimate the spread of datapoints for a given radius. It assumes a projection of the data onto a sphere of radius r, then performs a Monte Carlo sampling to estimate the area covered by the dataset. 

The `--point-radius` option determines whether the randomly choosen point is considered enclosed in the surface of the datapoint. The radius is considered on the surface, so the point must be within this point radius, traveling along the sphere. In other words, `--point-radius` is more similar to a great-circle distance cutoff.

Right now the the code only works on a full sphere, so the `--min-theta`, `--max-theta`, `--min-phi`, `--max-phi` should not be used as it *will produce incorrect results*.

# Data input

The code assumes a 3D set that can be loaded with `numpy.loadtxt`. If multiple filenames are given, it will stack into a single dataset.
