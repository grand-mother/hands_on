# Instructions for installing Python3.7

## First, check your Python version

- If Python is already available on your system it is likely to
  be `Python 2`. You can check your version as:
```bash
python --version
```

- If both `Python 2` and `Python 3` are already installed, you might need to
  explicitly call `python3` instead of `python`. Then, to check your `Python 3`
  version:
```bash
python3 --version
```

_If `Python 3.7` is already installed on your system, you can
skip these instructions._

---

## Installing Python 3.7 on **OSX**

On **OSX**, `Python3.7` can be installed with [homebrew][HOMEBREW], as:
```
brew unlink python
brew install python
```

[HOMEBREW]: https://github.com/Homebrew

---

## Installing Python 3.7 from the source

_If not available from your local package manager, `Python 3.7` can be installed
from the source._

1. Get the source distribution from [www.python.org][PYTHON_SOURCE] and extract
   it locally. E.g. as:
  ```
  wget https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tgz
  tar xzf Python-3.7.2.tgz
  ```

2. Configure, build and install (requires .red[root] privileges), E.g. as:
  ```
  cd Python-3.7.2
  ./configure
  sudo make altinstall
  ```
  > You'll get a warning that your Python build is not optimized. You can
  > ignore it for the present purpose.


[PYTHON_SOURCE]: https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tgz
