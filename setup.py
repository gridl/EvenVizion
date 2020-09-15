import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evenvizion",
    version="0.9.3",
    author="AIHunters",
    author_email="oss@aihunters.com",
    description="EvenVizion - is a video-based camera localization component",
    url="https://github.com/AIHunters/EvenVizion",
    long_description=long_description,
    long_description_content_type="text/markdown",	
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
	'imutils>=0.5.3',
	'numpy>=1.19.2',
	'opencv-python==3.4.2.17',
'opencv-contrib-python==3.4.2.17'],
    python_requires='>=3.6',
)

