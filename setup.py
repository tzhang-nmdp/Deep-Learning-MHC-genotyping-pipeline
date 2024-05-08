from setuptools import setup, find_packages

setup(
  name = 'kirai-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.5',
  license='MIT',
  description = 'kirAI - Pytorch',
  author = 'Tao Zhang',
  author_email = 'tzhang@nmdp.org',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/dohlee/kirai-pytorch',
  keywords = [
    'artificial intelligence',
    'genomics',
    'RNA splicing',
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
