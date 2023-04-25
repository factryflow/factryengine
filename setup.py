from setuptools import setup, find_packages

def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name='pyplanpro',
    version='0.0.1',
    author='Jacob Østergaard Nielsen',
    author_email='jaoe@oestergaard-as.dk',
    description='Task scheduler for python',
    long_description=open('README.md').read(),
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