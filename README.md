# RENNET
## Deep Learning Utilities for Audio Segmentation

RENNET is the library of useful classes, functions, parsers, etc., (and may be also application backend) for Audio Segmentation using Deep Learning.

> *ˈrɛnɪt*
>
> curdled milk from the stomach of an unweaned calf, containing rennin and used in curdling milk for cheese.

## Installation
At the moment, RENNET is not available on PyPI as a downloadable package.

Please install it from your local copy of this repository,
preferably in a local Python environment,
by running the following when your shell is open at the root of your copy of this repository:
```
pip install .
```

For installing extras, you can specify one or more options from the list below as, e.g. ```pip install -e .[analysis,dev]```
- `analysis` :: for data-analysis and preparation tasks alongside the package.
- `test` :: for testing the package.
- `dev` :: for development on the package (use `-e` to install it in editable form).

Please refer to `setup.py` to find the list of packages used by this library,
and follow their respective guidelines to install any specific versions of these libraries (e.g. to install a GPU enabled version of Tensorflow)

## Documentation
`TODO`

## Replicating [Double-Talk Detection Paper Submitted to Interspeech 2018]()
To replicate the experiments done in the full [thesis](http://publica.fraunhofer.de/documents/N-477004.html) behind the [Double-Talk detection paper submitted to Interspeech 2018](), please follow the instructions in `examples/double-talk-detection/README.md`.

## License
Apache License 2.0
