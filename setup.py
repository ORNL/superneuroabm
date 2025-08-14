from setuptools import setup, find_packages

setup(
    name="SuperNeuroABM",
    version="1.0.0.dev1",
    author="Chathika Gunaratne, Shruti Kulkarni, Ashish Gautam, Xi Zhang, Prasanna Date",
    author_email="gunaratnecs@ornl.gov",
    packages=[
        "superneuroabm",
        "superneuroabm.step_functions",
        "superneuroabm.step_functions.soma",
        "superneuroabm.step_functions.synapse",
        "superneuroabm.step_functions.synapse.stdp",
    ],
    include_package_data=True,
    url="https://code.ornl.gov/superneuro/superneuroabm",
    license="GPL",
    description="A GPU-based multi-agent simulation framework for neuromorphic computing.",
    long_description="""A GPU-based multi-agent simulation framework for neuromorphic computing.""",
    long_description_content_type="text/markdown",
    project_urls={"Source": "https://github.com/ORNL/superneuroabm"},
    install_requires=[
        "sagesim",
    ],
)
