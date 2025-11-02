from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="STmulator",
    version="0.1.6",
    author="Yiru CHEN",
    author_email="yiru.22@intl.zju.edu.cn",
    description="Spatial Transcriptomics Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maiziezhoulab/SimulaTor",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "alphashape>=1.3.1",
        "anndata>=0.9.2",
        "geomloss>=0.2.6",
        "joblib>=1.4.2",
        "matplotlib>=3.7.5",
        "numba>=0.55.2",
        "numpy>=1.22.4",
        "pandas>=2.0.3",
        "paste2>=1.0.1",
        "scanpy>=1.9.8",
        "scikit-learn>=1.2.1",  
        "scipy>=1.10.1",
        "seaborn>=0.13.2",  
        "torch>=1.12.1",
        "tqdm>=4.66.4",
    ],
)