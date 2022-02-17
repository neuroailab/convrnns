from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="convrnns",
    version="1.0",
    description="Convolutional recurrent neural networks (ConvRNNs) pretrained on ImageNet",
    long_description=readme,
    author="Aran Nayebi",
    author_email="anayebi@stanford.edu",
    url="https://github.com/neuroailab/convrnns",
    packages=find_packages(),
    install_requires=["networkx==1.11", "tensorflow==1.13.1", "numpy>=1.15.0"],
    python_requires=">=2.7",
    license="MIT license",
    keywords="convrnns",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
    ],
)
