from setuptools import Extension, find_packages, setup


setup(
    name="cloud_rad_scheme",
    version="0.0.0",
    author="J. Feng",
    author_email="",
    description="Cloud optics library.",
    url="",
    python_requires=">=3.6",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: ",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "miepython",
        "netCDF4",
        "numpy",
        "scipy",
    ],
)
