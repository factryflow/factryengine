import setuptools

import versioneer

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="planbee",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="production / job shop / resource scheduler for Python",
    author="Jacob Østergaard Nielsen",
    author_email="jaoe@oestergaard-as.dk",
    url="https://github.com/Yacobolo/PlanBee",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "testing": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)
