from setuptools import setup, find_packages

contrib = [
    'Raphael Ortiz',
]

# setup.
setup(name='goid',
      version='0.1.1',
      description='Analysis of gastruloid images',
      author=', '.join(contrib),
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'scikit-image',
          'imagecodecs',
          'tqdm',
          'pytest',
          'luigi',
          'openpyxl',
          'dl-utils @ git+https://github.com/fmi-basel/dl-utils',
          'improc @ git+https://github.com/fmi-basel/improc',
          'inter_view @ git+https://github.com/fmi-basel/inter-view',
      ],
      entry_points={
          'console_scripts': [
              'goid=goid.luigi_workflow:main',
              'goid_train_fg=goid.foreground_model.train:main',
              'goid_train_debris=goid.debris_model.train:main',
              'goid_train_separator=goid.separator_model.train:main',
          ],
      },
      zip_safe=False)
