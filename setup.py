from setuptools import setup, find_packages

setup(
    name="SuperNeuroABM",
    version="0.2.0",
    author="Chathika Gunaratne, Prasanna Date, Shruti Kulkarni, Xi Zhang",
    author_email="gunaratnecs@ornl.gov",
    packages=["superneuroabm", "superneuroabm.core"],
    include_package_data=True,
    url="https://code.ornl.gov/superneuro/superneuroabm",
    license="GPL",
    description="A GPU-based multi-agent simulation framework for neuromorphic computing.",
    long_description="""A GPU-based multi-agent simulation framework for neuromorphic computing.""",
    long_description_content_type="text/markdown",
    project_urls={"Source": "https://github.com/ORNL/superneuroabm"},
    install_requires=["numba==0.55.1", "numpy==1.21.6", "tqdm==4.64.1"],
)
