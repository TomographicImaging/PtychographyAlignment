# PtychographyAlignment
The [`applications`](applications) directory contains Jupiter notebooks to analyse experimental data provided by Diamond. Edit [`paths`](src/config/paths.py) and [`user_paths`](src/config/user_paths.py) to configure the filepaths for your data.
The [`converters`](converters) directory contains scripts to convert the data/files between different formats.
The [`src`](src) contains the core code for the package. In particular, the code is structured as follows:
- [`config`](src/config/) contains configurations settings for the code, such as the data filepaths.
- [`CT reconstructions`](src/CT_reconstruction/) tomographic reconstructions methods
- [`Initial_code_by_Diamond`](src/Initial_code_by_Diamond/)
- [`io_module`](src/io_module/)
- [`quality_metrics`](src\quality_metrics)
- [`simulations`](src/simulations/) contains the tools to simulate tomographic projections with [gVXR](https://gvirtualxray.sourceforge.io/), where each projections is shifted along the "x" and "y" axes, i.e., in the transverse plane to the x-rays propagation direction.
- [`utilities`](src/utilities/)
- [`validating_methods`](src/validating_methods/)
- [`viewer`](src/viewer/)

The [`tests`](tests) folder contains (some) tests of the methods developed, whereby [`helpers`](tests/helpers) contains small utility functions or classes for tests. 

Note: the code is work in progress. 

## Contributing
The [environment file](environment.yml) contains the dependencies needed to run the package.

To install locally allowing edits:
```pip install -e .```

To run tests:
```pytest```