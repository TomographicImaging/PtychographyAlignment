# PtychographyAlignment
The [`applications`](applications) directory contains Jupiter notebooks to analyse experimental data provided by Diamond. Edit [`paths`](src\config\paths.py) and [`user_paths`](src\config\user_paths.py) to configure the filepaths for your data.
The [`converters`](converters) directory contains scripts to convert the data/files between different formats.
The [`src`](src) contains the core code for the package.
The [`tests`](tests) contains some tests of the methods developed, whereby [`helpers`](tests\helpers) contains small utility functions or classes for tests. 

Note: the code is work in progress. 

## Contributing
The [environment file](environment.yml) contains the dependencies needed to run the package.

To install locally allowing edits:
```pip install -e .```

To run tests:
```pytest```