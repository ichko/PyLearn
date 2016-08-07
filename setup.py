from setuptools import setup, find_packages

setup(
        name="PyLearn",
        version="0.1",
        packages=find_packages(),
        install_requires=['numpy>=1.11.1', 'matplotlib>=1.5.1'],
        author="Iliya Zhechev",
        author_email="ichko@habala.babala",
        description="Machine Learning Lib",
        license="PSF",
        keywords="Ml machine learning",
        url="https://github.com/ichko/PyLearn"
    )
