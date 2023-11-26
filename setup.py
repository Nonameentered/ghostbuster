from setuptools import setup, find_packages

setup(
    name="ghostbuster",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'scikit-learn',
        'numpy',
        'tenacity',
        'openai==0.28.1',
        'torch',
        'tiktoken',
        'flask',
        'tabulate',
        'dill',
        'nltk'
    ],
    package_data={
        'ghostbuster': ['model/features.txt', 'model/model', 'model/mu', 'model/sigma']
    },
    include_package_data=True,
)
