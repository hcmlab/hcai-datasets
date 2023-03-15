"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import datetime
import sys
import pkg_resources
import pathlib
from version import __version__

# Parse arguments
if "--nightly" in sys.argv:
    nightly = True
    sys.argv.remove("--nightly")
else:
    nightly = False

# Settings
project_name = "hcai-datasets"
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


# Adjustment for nightly build
if nightly:
    project_name += "-nightly"
    datestring = datetime.datetime.now().strftime("%Y%m%d%H%M")
    curr_version = pkg_resources.parse_version(__version__)
    __version__ = f"{curr_version.base_version}.dev{datestring}"

# Setup
setup(
    name=project_name,
    version=__version__,
    description="!Alpha Version! - This repository contains the backend server for the nova annotation ui (https://github.com/hcmlab/nova)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hcmlab/nova-server",
    author="Dominik Schiller",
    author_email="dominik.schiller@uni-a.de",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6, <4",
    install_requires=[
        "pymongo==3.12",
        "decord;platform_machine=='AMD64'",
        "numba",
        "pandas==1.4"
    ],
    extras_require={
        "torch": ["torch"],
        "tf" : [  "tensorflow==2.7", "tensorflow-datasets==4.4.0"]
    }
)
