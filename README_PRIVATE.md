### Building Pypi package ###

build: ```python -m build```

upload test: ```python -m twine upload --repository testpypi dist/*```

upload: ```python -m twine upload dist/*```

username = `__token__`

password = <the token value, including the `pypi-` prefix>


### Generate Requirements ###
pipreqs . --force
 

### Installing tensorflow-dataset-validation ### 
If tensorflow-dataset-validation cannot be installed du to pypi being stuck in dependency backtracking use
```pip install --use-feature=fast-deps --use-deprecated=legacy-resolver tensorflow-dataset-validation```


See [here](https://github.com/pypa/pip/issues/9187) for details.


