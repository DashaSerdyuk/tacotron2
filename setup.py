from pathlib import Path
from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def get_version(filename):
    from re import findall
    with open(filename) as f:
        metadata = dict(findall("__([a-z]+)__ = '([^']+)'", f.read()))
    return metadata['version']


setup(
    name='tacotron2',
    version=get_version('tacotron2/__init__.py'),
    description='Tacotron2 TTS Model',
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=[
        "torch==1.3.1",
        "scikit-learn==0.21.3",
        "pandas==0.25.1",
        "future==0.17.1",
        "tqdm==4.36.1",
        "librosa==0.7.2",
        "pillow==7.0.0",
        "matplotlib==3.1.3",
        "num2words==0.5.10",
        "rnnmorph==0.4.0",
        "lxml==4.5.0",
        "transformers==2.4.1",
        "russian_g2p @ http://github.com/nsu-ai/russian_g2p/tarball/master",
        "dawg @ http://github.com/pytries/DAWG/tarball/master"
    ],
    package_dir={'tacotron2': 'tacotron2'},
    package_data={'tacotron2': ['tokenizers/data/*']}
)
