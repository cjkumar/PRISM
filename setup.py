"""PRISM package setup."""

from setuptools import setup, find_packages
from pathlib import Path

readme = ""
requirements = []
req_file = Path(__file__).parent / "requirements.txt"
if req_file.exists():
    requirements = [
        line.strip()
        for line in req_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="prism-policy",
    version="1.0.0",
    description=(
        "PRISM - Policy Reasoning Integrated Sequential Model. "
        "Multi-Agent AI for National Disease Control Plan Analysis."
    ),
    author="Health Systems Innovation Lab",
    author_email="",
    url="https://doi.org/10.7910/DVN/ETVLMD",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "full": [
            "transformers>=4.37.0",
            "accelerate>=0.25.0",
            "opencv-python>=4.8.0",
            "pdf2image>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prism=PRISM.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
