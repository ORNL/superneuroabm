from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="superneuroabm",
    version="1.0.0",
    author="Chathika Gunaratne, Shruti Kulkarni, Ashish Gautam, Xi Zhang, Prasanna Date",
    author_email="gunaratnecs@ornl.gov",
    packages=[
        "superneuroabm",
        "superneuroabm.step_functions",
        "superneuroabm.step_functions.soma",
        "superneuroabm.step_functions.synapse",
        "superneuroabm.step_functions.synapse.stdp",
        "superneuroabm.io",
    ],
    package_data={
        "superneuroabm": ["*.yaml"],
    },
    include_package_data=True,
    url="https://github.com/ORNL/superneuroabm",
    license="BSD-3-Clause",
    description="A GPU-based multi-agent simulation framework for neuromorphic computing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/ORNL/superneuroabm",
        "Bug Tracker": "https://github.com/ORNL/superneuroabm/issues",
    },
    install_requires=["sagesim==0.5.0", "pyyaml", "networkx", "matplotlib"],
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

