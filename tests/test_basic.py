import importlib
import pathlib


def test_package_importable():
    """Ensure the bcpc package can be imported without side-effects."""
    pkg = importlib.import_module("src")
    assert hasattr(pkg, "logger")


def test_project_paths_exist():
    """Verify core project directories exist or are created at runtime."""
    pkg = importlib.import_module("src")
    for p in (pkg.DATA_DIR, pkg.INPUT_DIR, pkg.OUTPUT_DIR):
        assert pathlib.Path(p).is_dir()
