# starspot-backlights
A suite of cartoon tools for modeling the effect of starspots on transiting exoplanet observations.

## Documentation
For detailed usage, please read the documentation [here](???).

## Example Usage
```
(include example)
```

## Installation
You should be able to install this by running
```
pip install starspot-backlights
```
from a UNIX/conda prompt.

If you want to be able to modify the code yourself, please also feel free to fork/clone this repository onto your own computer and install directly from that editable package. For example, this might look like:
```
git clone https://github.com/zkbt/starspot-backlights.git
cd starspot-backlights
pip install -e .
```
This will link the installed version of the `starspotbacklights` package to your local repository. Changes you make to the code in the repository should be reflected in the version Python sees when it tries to `import starspotbacklights`.

## Contributors

This package was written mostly by [Zach Berta-Thompson](https://github.com/zkbt). It builds on the cartoon starspot model presented in Berta et al. (2012) but more closely follows the transit light source effect framework described in Rackham et al. (2018, 2019).
