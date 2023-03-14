import ast
from pathlib import Path
from importlib.util import find_spec


def get_imports(file_path):
    """Returns a list of imported packages for the respective module"""
    with open(file_path, "r") as f:
        node = ast.parse(f.read())
    imports = [n.names[0].name for n in node.body if isinstance(n, ast.Import)]
    return imports


def validate_installation(file_path: Path):
    """Checks if every dependency in the list is installed"""
    dependencies = get_imports(file_path)
    module_id = file_path.name[:-3]

    for dependency in dependencies:
        if not find_spec(dependency):
            print(f"Package {dependency} not installed. Import of {module_id} skipped")
            return False
    return True
