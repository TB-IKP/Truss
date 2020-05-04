from setuptools import setup

setup(
        name='Truss',
        version='1.0.0',
        description='Optimal truss structures using the Augmented Lagrangian Method',
        url='',
        author='Tobias Beck',
        author_email='becktobias@gmx.net',
        #license=None,
        python_requires='>=3',
        packages=['Truss'],
        install_requires=['numpy','scipy','matplotlib','numdifftools','ipopt'],
)