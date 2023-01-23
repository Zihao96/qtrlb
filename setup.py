from setuptools import setup

setup(name='qtrlb',
      version='0.1.0',
      description='Quantum Control using Qblox',
      url='https://github.com/Zihao96/qtrlb.git',
      author='Zihao Wang, Rayleigh Parker, Machiel Blok',
      author_email='zwang156@ur.rochester.edu',
      #license='MIT',
      install_requires=['qblox-instruments', 'matplotlib'],
      packages=['qtrlb'],
      python_requires=">=3.8",
      zip_safe=False)