# PtychographyAlignment
The [`applications`](applications) directory contains Jupiter notebooks to analyse experimental data provided by Diamond. Edit [`paths`](src/config/paths.py) and [`user_paths`](src/config/user_paths.py) to configure the filepaths for your data.
The [`converters`](converters) directory contains scripts to convert the data/files between different formats.
The [`src`](src) contains the core code for the package. In particular, the code is structured as follows:
- [`alignment`](src/alignment/)
- [`config`](src/config/) contains configurations settings for the code, such as the data filepaths.
- [`CT reconstructions`](src/CT_reconstruction/) tomographic reconstructions methods
- [`io_module`](src/io_module/)
- [`pipeline`](src/pipeline/)
- [`quality_metrics`](src/quality_metrics)
- [`recipe`](recipe) contains environment file
- [`simulations`](src/simulations/) contains the tools to simulate tomographic projections with [gVXR](https://gvirtualxray.sourceforge.io/), where each projections is shifted along the "x" and "y" axes, i.e., in the transverse plane to the x-rays propagation direction.
- [`utilities`](src/utilities/)
- [`validating_methods`](src/validating_methods/)
- [`viewer`](src/viewer/)

The [`tests`](tests) folder contains (some) tests of the methods developed, whereby [`helpers`](tests/helpers) contains small utility functions or classes for tests. 

Note: the code is work in progress. 

## Contributing

### Installation

Develop code locally by cloning the source code, creating a development environment and installing it.

1. Install [miniconda](https://repo.anaconda.com/miniconda/), then launch the `Miniconda Prompt`.

2. Clone the `main` branch of `PtychographyAlignment` locally, and navigate into where it has been cloned:
```sh
git clone git@github.com:TomographicImaging/PtychographyAlignment.git
cd PtychographyAlignment
```

3. The [environment file](recipe/environment.yml) contains the dependencies needed to run the package.
Create the conda environment using the following command:
```sh
conda env create -f recipe/environment.yml
```

4. Activate the environment:
```sh
conda activate ptychotomo_env
```

5. Install the package:
```sh
pip install .
```

or

Install the package locally allowing edits:
```sh
pip install -e .
```
### Testing

6. Tests require the data "pollen_Volpe", please store this as specified in [`paths`](src/config/paths.py). To run tests:
```sh
pytest
```

Test can run with visualisation outputs from the ccpi-viewer:
```sh
pytest --viewer
```

### Conventions

For the docstrings, please use the [numpydoc convention](https://numpydoc.readthedocs.io/en/latest/format.html).

Example:
```
"""
Summarize the function in one line.

Several sentences providing an extended description. Refer to
variables using back-ticks, e.g. `var`.

Parameters
----------
var1 : array_like
    Array_like means all those objects -- lists, nested lists, etc. --
    that can be converted to an array.  We can also refer to
    variables like `var1`.
var2 : int
    The type above can either refer to an actual Python type
    (e.g. ``int``), or describe the type of the variable in more
    detail, e.g. ``(N,) ndarray`` or ``array_like``.
*args : iterable
    Other arguments.
long_var_name : {'hi', 'ho'}, optional
    Choices in brackets, default first when optional.

Returns
-------
type
    Explanation of anonymous return value of type ``type``.
describe : type
    Explanation of return value named `describe`.
out : type
    Explanation of `out`.
type_without_description
"""
```
