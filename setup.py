from distutils.core import setup

setup(
    name='rrl',
    url='http://www.igsor.net/research/rrl/',
    author='Matthias Baumgartner',
    author_email='research@igsor.net',
    version='1.0',
    packages=['rrl'],
    license='Free for use',
    long_description=open('README').read(),
    requires=("scipy","numpy","mdp","Oger")
)
