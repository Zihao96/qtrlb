from setuptools import setup
from platform import platform

install_requires = [
    'numpy',
    'scipy', 
    'matplotlib',
    'lmfit',
    'ruamel.yaml<0.18.0',
    'pyvisa',
    'qblox-instruments==0.10.0',
    'scikit-learn==1.6.1',
    'qiskit==1.3.2',
    # 'scqubits',
]

if platform().startswith('macOS'): install_requires.append('pyvisa-py')


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
