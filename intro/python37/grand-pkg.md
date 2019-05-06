# GRAND package manager

> In the following, when it is stated `pip3 install` it must be understood that
> `pip3` relates to `Python 3.7`. The corresponding `pip` command might differ
> on your system. Then modify the examples accordingly.

---

## 1. Installation of the GRAND package manager

The GRAND package manager is available from
[PyPi](https://pypi.org/project/grand-pkg). It can be installed as:
```
pip3 install --user grand-pkg
```
The source are on [GitHub](https://github.com/grand-mother/pkg).

> If you find a bug, have troubles or new suggestions concerning the package
> manager, you can open a new
> [issue](https://github.com/grand-mother/pkg/issues).

_Once properly installed you should now have three new commands:
`grand-pkg-init`, `grand-pkg-update` and `grand-pkg-config`._

---

## 2. Creation of a new GRAND package

Create a new folder, e.g.: `mypackage`. Then **inside** this new folder simply
call:
```
grand-pkg-init
```
Follow the instructions and answer the questions.
> You can skip any question by just typing `<enter>`.

Your folder should now be populated with several files and folders. Many of them
are for administrative purpose, e.g. web services through GitHub, licensing and
distributing. You wont need to modify them for now. The relevant elements are:

- ### mypackage/
  The source of your `Python` package. It comes with a default `__init__.py`
  loading version & git meta data.

- ### docs/
  The documentation of your package is expected to be located there. It starts
  with a default `README.md`.

- ### tests/
  Unit tests of your package are expected to be located there. It starts with
  a pre-configured environment based on the `unittest` and `doctest` packages.

---

## 3. Extending your package

- Add a new sub-module to your package with a function taking one argument,
   `name`, and returning `f"Hello {name}!"`. Your function should raise a
   `ValueError` if name equals `Olivier`.
  > GRAND packages follow the [PEP8][PEP8] coding style. You can reformat your
  > code using the autopep8 utility, installed with `grand-pkg`.

- Document your function using [numpy docstring style][NPY_DOCSTRING].
  > You **must** document the input argument, the function return value and the
  > exception it can raise. In addition you should include at least one
  > **example** of usage in the `Examples` section, following the `doctest`
  > syntax.

- Add and commit your changes.
  > When committing your changes you **should see** a message from `grand-pkg`
  > inside the commit log.

[PEP8]: https://www.python.org/dev/peps/pep-0008
[NPY_DOCSTRING]: https://numpydoc.readthedocs.io/en/latest/format.html

---

## 4. Testing your package

- The test suite can be ran as:
  ```
  python3 -m tests
  ```

- Add a unit test for your new sub-module and its function.
  > You can copy the `tests/test_version.py` and rename it to
  > `tests/test_mymodule.py`. Note that any file starting as `tests/test_*`.

---

## 5. Modifying the package meta-data

The `grand-pkg-config` command allows to print, edit and modify the package
meta-data: name, description, etc. For example you can try:
```
grand-pkg-config --edit
```
