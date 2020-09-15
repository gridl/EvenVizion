import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EvenVizion-pkg-AIHunters",
    version="0.9",
    author="AIHunters",
    author_email="oss@aihunters.com",
    description="EvenVizion - is a video-based camera localization component",
    url="https://github.com/AIHunters/EvenVizion",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
