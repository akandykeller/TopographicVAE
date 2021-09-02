from setuptools import setup

setup(
    name='TVAE',
    version='0.0.1',
    description="TVAE",
    author="T. Anderson Keller",
    author_email='',
    packages=[
        'tvae'
    ],
    entry_points={
        'console_scripts': [
            'tvae=tvae.cli:main',
        ]
    },
    python_requires='>=3.6',
)
