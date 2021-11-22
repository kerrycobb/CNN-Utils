
import setuptools as st

st.setup(
    name='CNN-Utils',
    version='1.0.0',
    description='Utility functions for doing convolutional neural network analysis with DNA sequence data',
    url='http://github.com/kerrycobb/CNN-Utils',
    author='Kerry A. Cobb',
    author_email='cobbkerry@gmail.com',
    license='MIT',
    packages=st.find_packages(exclude=["tests*"]),
#    install_requires=[
#        'numpy',
#        'sklearn']
)
