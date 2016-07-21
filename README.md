# RT4Mars

<<<<<<< HEAD
![Badge Status](https://ci-as-a-service)

=======
>>>>>>> 6c43430ffdcfc86fe3b997c71bffdff954633d69
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
    IPython$ rt = RT(molelist)
    IPython$ rt.get_abscoef()
    ```

4. To compute a Brightness temperature
    ```console
    IPython$ Tb = rt.radiative_transfer()
    ```

5. To see the settings
<<<<<<< HEAD
```console
IPython$ print(rt)
```
=======
    ```console
    IPython$ print(rt)
    ```
>>>>>>> 6c43430ffdcfc86fe3b997c71bffdff954633d69

## Installation

    $ git clone https://github.com/kazu41/RT4Mars

## Author

[Kazutoshi Sagi](kazutoshi.sagi@gmail.com "gmail")

## License
