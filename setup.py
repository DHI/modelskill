import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE") as fh:
    license = fh.read()

setuptools.setup(
    name="fmskill",
    version="0.8.0",
    install_requires=[
        "numpy",
        "pandas",
        "mikeio >= 1.2",
        "matplotlib",
        "xarray",
        "markdown",
        "jinja2",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx==4.5.0",  # pin version to work with sphinx-book-theme
            "sphinx-book-theme",
            "black==22.3.0",
            "plotly >= 4.5",
        ],
        "test": [
            "pytest",
            "netCDF4",
            "openpyxl",
            "dask",
        ],
        "notebooks": [
            "nbformat",
            "nbconvert",
            "jupyter",
            "plotly",
            "shapely",
        ],
    },
    entry_points="""
        [console_scripts]
            fmskill=fmskill.cli:report
    """,
    author="Jesper Sandvig Mariegaard",
    author_email="jem@dhigroup.com",
    description="Compare results from MIKE simulations with observations.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/fmskill",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
)
