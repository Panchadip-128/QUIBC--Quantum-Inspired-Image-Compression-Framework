"""
setup.py for QUIBC
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("umap")
    ]

setup(
    name="quibc",
    version="1.0.0",
    author=(
        "Panchadip Bhattacharjee, Somyajeet Arukh, Arya Abnish Singh, "
        "Jonath Jimmi, Gururaj H L"
    ),
    author_email="panchadip.mitblr2023@learner.manipal.edu",
    description=(
        "QUIBC: A Quantum-Inspired Image Binarization Compressor "
        "for Resource-Constrained Edge Devices"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Panchadip-128/QUIBC",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=(
        "quantum-inspired image-compression edge-ai binarization "
        "rate-distortion iot deep-learning tensorflow"
    ),
    entry_points={
        "console_scripts": [
            "quibc-train=scripts.train:main",
            "quibc-eval=scripts.evaluate:main",
            "quibc-compress=scripts.compress:main",
        ]
    },
    include_package_data=True,
)
