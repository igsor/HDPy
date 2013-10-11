from distutils.core import setup

setup(
    name='HDPy',
    url='http://www.igsor.net/research/HDPy/',
    author='Matthias Baumgartner',
    author_email='research@igsor.net',
    version='1.0',
    packages=['HDPy'],
    license='Free for use',
    long_description=open('README').read(),
    requires=("scipy","numpy","mdp","Oger")
)
