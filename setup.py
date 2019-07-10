from setuptools import setup

setup(
    name = "CALC",
    version = "0.1",
    author = "Bryan Tripp",
    author_email = "bptripp@uwaterloo.ca",
    description = ("Convolutional architecture like cortex."),
    license = "BSD",
    keywords = "primate vision convolutional network",
    url = "https://github.com/bptripp/calc",
    packages=['calc', 'calc.examples'],
)

