# RT4Mars

=======
Radiative transfer codes for the Martian atmosphere and a sub-mm frequency region

## Description
`RT4Mars` is a python code set for the Martian atmosphere and a sub-mm frequency region. This allows us to compute an absorption coefficient and Brightness temperatures.

## Features

- atmospheric emission and absorption
- no scattering for the moment
- onion peeling approximation

## Requirement

- Python 2.7
- IPython
- matplotlib
- numpy
- scipy

## Usage

1. Open IPython
    ```console
    $ ipython --pylab
    ```

2. execute main.py
    ```console
    IPython$ %run main.py
    ```

3. To generate Absorption coefficient
    ```console
    IPython$ rt = RT('where/you/put/ctlfile.py')
    IPython$ rt.get_abscoef()
    ```

4. To compute a Brightness temperature
    ```console
    IPython$ Tb = rt.radiative_transfer()
    ```

5. To see the settings
    ```console
    IPython$ print(rt)
    ```

## Installation
download
[the latest ver.](https://github.com/kazu41/RT4Mars/archive/master.zip)

or

    $ git clone https://github.com/kazu41/RT4Mars

## Author

Kazutoshi Sagi -- kazutoshi.sagi@gmail.com

## License
