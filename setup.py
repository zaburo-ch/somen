from setuptools import find_packages, setup

setup(
    name="somen",
    version="0.0.0",
    packages=find_packages("."),
    install_requires=[
        "pandas",
        "numpy",
        "tables",
        "torch",
        "scikit-learn",
        "pfio",
        "albumentations",
        "pytorch-pfn-extras",
        "torchvision",
        "feather-format",
        "dacite",
        "colorlog",
        "tqdm",
    ],
    package_data={"somen": ["py.typed"], "somen.logger": ["_default_logging_config.yaml"]},
)
