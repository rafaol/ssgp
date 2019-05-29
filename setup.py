import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('LICENSE') as f:
    license = f.read()


setuptools.setup(
    name="ssgp",
    version="0.0.1",
    author="Rafael Oliveira",
    author_email="rafael.oliveira@sydney.edu.au",
    description="An implementation of sparse spectrum Gaussian processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rafaol/ssgp",
    packages=setuptools.find_packages(exclude=('notebooks','docs')),
    license=license,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

