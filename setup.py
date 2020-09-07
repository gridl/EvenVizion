import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EvenVizion-pkg-Oxagile_Rnd",
    version="0.0.7",
    author="Rnd_Oxagile",
    author_email="rd@oxagile.com.",
    description="EvenVizion - is a video-based camera localization component",
    url="https://github.com/RnD-Oxagile/EvenVizion",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)