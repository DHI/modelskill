import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE") as fh:
    license = fh.read()

setuptools.setup(
    name="mikefm_skill",
    version='0.1.dev',
    install_requires=["numpy", "pandas", "scipy", "mikeio", "shapely"],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "sphinx",
            "sphinx-rtd-theme",
            "matplotlib",
            "jupyterlab",
        ],
        "test": ["pytest", "matplotlib"],
    },
    author="Henrik Andersson",
    author_email="jan@dhigroup.com",
    description="Compare results from MIKE FM simulations with observations.",
    platform="windows_x64",
    license=license,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/mikefm-skill",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
    ],
)
