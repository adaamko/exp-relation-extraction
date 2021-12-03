from setuptools import find_packages, setup

setup(
    name="xpotato",
    version="0.0.2",
    description="XAI human-in-the-loop information extraction framework",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adaamko/POTATO",
    author="Adam Kovacs, Gabor Recski",
    author_email="adam.kovacs@tuwien.ac.at, gabor.recski@tuwien.ac.at",
    license="MIT",
    install_requires=[
        "beautifulsoup4",
        "tinydb",
        "pandas",
        "tqdm",
        "stanza",
        "sklearn",
        "eli5",
        "matplotlib",
        "graphviz",
        "openpyxl",
        "penman",
        "networkx",
        "rank_bm25",
        "streamlit",
        "streamlit-aggrid",
        "scikit-criteria",
        "tuw-nlp",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
