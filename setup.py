from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'general solutions'
LONG_DESCRIPTION = 'function to get general solutions to nonhomogeneous linear equations, Ax = b'

# Setting up
setup(
        name="general_solutions", 
        version=VERSION,
        author="Wang Mengchang",
        author_email="<wangmengchang@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy" ],
        keywords=['python', 'general solutions'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
