from pkg_resources import parse_requirements
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# requirements list :
with open('requirements.txt') as f:
    requirements = [str(req) for req in parse_requirements(f)]

setup(
    name="OOF",
    version="1.0.0",
    description="Out of field dose estimation using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maichi98/OOF",
    author="EL AICHI MOHAMMED",
    author_email="mohammed.el-aichi@gustaveroussy.fr",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(),  # Required
    install_requires=requirements,

)
