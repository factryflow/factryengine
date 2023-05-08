import io
import os
from setuptools import setup, find_packages

def read(*paths, **kwargs):
    """Read the contents of a text file safely."""
    
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name='pyplanpro',
    version='0.0.6',
    author='Jacob Østergaard Nielsen',
    author_email='jaoe@oestergaard-as.dk',
    description='Task scheduler for python',
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/Oestergaard-A-S/PyPlanPro',
    install_requires= read_requirements("requirements.txt"),
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    
)
