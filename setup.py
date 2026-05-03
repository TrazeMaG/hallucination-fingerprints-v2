from setuptools import setup, find_packages

setup(
    name="hallscope",
    version="0.1.0",
    author="Nikhil Upadhyay",
    author_email="nikhil25000@gmail.com",
    description="Hallucination interpretability library for transformer language models",
    long_description=open("README_HALLSCOPE.md").read() if __import__("os").path.exists("README_HALLSCOPE.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/TrazeMaG/hallucination-fingerprints-v2",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformer_lens>=2.0.0",
        "transformers>=4.40.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)