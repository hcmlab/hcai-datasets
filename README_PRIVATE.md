build: ```python -m build```

upload test: ```python -m twine upload --repository testpypi dist/*```

upload: ```python -m twine upload dist/*```

username = `__token__`

password = <the token value, including the `pypi-` prefix>