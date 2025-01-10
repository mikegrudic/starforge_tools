import setuptools, os
from setuptools import setup


thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(name='starforge_tools',
      version='0.1',
      description='Package for analysis of STARFORGE simulations',
      url='http://github.com/mikegrudic/starforge_tools',
      author='Mike Grudić',
      author_email='mike.grudich@gmail.com',
      license='MIT',
#      packages=['meshoid'],
      project_urls={
        "Bug Tracker": "https://github.com/mikegrudic/starforge_tools",
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      package_dir={"": "src"},
      packages=setuptools.find_packages(where="src"),
      python_requires=">=3.9",
      zip_safe=False,
      install_requires=install_requires
     )
