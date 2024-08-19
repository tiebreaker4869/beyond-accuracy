from setuptools import setup, find_packages

setup(
    name="beyond_accuracy",
    version="1.0.0",
    author="Lifan Sun",
    author_email="lifansun1412@gmail.com",
    description="A Recommendation System Evaluation Toolkit beyond accuracy metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tiebreaker4869/beyond-accuracy",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn'
    ],
    )
