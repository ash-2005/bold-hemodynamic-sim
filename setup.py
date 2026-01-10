"""setup.py for bold-hemodynamic-sim."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

setup(
    name="bold-hemodynamic-sim",
    version="0.4.0",
    author="Ashmit Gupta",
    description=(
        "Standalone Balloon-Windkessel haemodynamic simulator "
        "and parameter analysis toolkit (Friston 2003). No TVB dependency."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ash-2005/bold-hemodynamic-sim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={"console_scripts": ["bold-sim=cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords=["fMRI", "BOLD", "haemodynamics", "balloon-windkessel", "neuroscience"],
)
