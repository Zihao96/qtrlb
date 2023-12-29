from setuptools import setup
from platform import platform

install_requires = [
    'qblox-instruments>=0.10.0',
    'scikit-learn',
    'scipy==1.10.0',
    'matplotlib==3.6.3',
    'lmfit==1.1.0',
    'scqubits==3.1.0',
    'qiskit==0.42.1',
    'cirq==1.1.0',
    'pyvisa==1.13.0'
]

if platform().startswith('macOS'): install_requires.append('pyvisa-py==0.7.0')


setup(name='qtrlb',
      version='0.1.0',
      description='Quantum Control using Qblox',
      url='https://github.com/Zihao96/qtrlb.git',
      author='Zihao Wang, Rayleigh Parker, Elizabeth Champion, Machiel Blok',
      author_email='zwang156@ur.rochester.edu',
      license='MIT',
      install_requires=install_requires,
      packages=['qtrlb'],
      python_requires=">=3.10.0",
      zip_safe=False)
