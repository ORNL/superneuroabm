from setuptools import setup

setup(
    name="SuperNeuroABM",
    version="0.1.0",
    author="Chathika Gunaratne, Prasanna Date, Shruti Kulkarni, Robert M. Patton, Mark Coletti",
    author_email="gunaratnecs@ornl.gov",
    packages=["superneuro"],
    include_package_data=True,
    url="https://github.com/ORNL/superneuroabm",
    license="BSD 3-Clause",
    description="A GPU-based multi-agent simulation framework for neuromorphic computing.",
    long_description="""A GPU-based multi-agent simulation framework for neuromorphic computing.""",
    long_description_content_type="text/markdown",
    project_urls={"Source": "https://github.com/ORNL/superneuroabm"},
    install_requires=["numba==0.55.1", "numpy==1.21.6", "tqdm==4.64.1"],
)
