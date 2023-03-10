from setuptools import setup, find_packages

setup(
    name="vandy_taggers",
    version="0.0.1",
    packages=find_packages(),
    description="Alternative package to fit and study particle taggers for CMS",
    install_requires=["torch"],
)
