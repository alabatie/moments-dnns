from setuptools import setup, find_packages

setup(
    name="moments-dnns",
    url="https://github.com/alabatie/moments-dnns",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "protobuf",
        "pyyaml",
        "tensorflow",
        "silence_tensorflow",
        "tqdm",
        "jupyterlab",
        "seaborn",
        "fire",
        "cython",
    ],
)
