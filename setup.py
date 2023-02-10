from setuptools import setup

setup(name='qtrlb',
      version='0.1.0',
      description='Quantum Control using Qblox',
      url='https://github.com/Zihao96/qtrlb.git',
      author='Zihao Wang, Rayleigh Parker, Machiel Blok',
      author_email='zwang156@ur.rochester.edu',
      #license='MIT',
      install_requires=['spyder==5.4.2','qblox-instruments==0.8.2','scipy==1.10.0', 'matplotlib==3.6.3', 'scikit-learn==1.2.1'],
      packages=['qtrlb'],
      python_requires="==3.11.0",
      zip_safe=False)
