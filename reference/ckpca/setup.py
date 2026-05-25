from setuptools import setup, find_packages

setup(
    name='ckpca',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'joblib',
        'tqdm',
        #'cupy',
        #'numba',
    ],
    entry_points={
        'console_scripts': [
            'ckpca-example=example:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['data/*', 'omega_cached/*'],
    },
    author='Lukas Koerber',
    author_email='koerber.lukas@icloud.com',
    description='CK-PCA: Characteristics Kernel Principal Component Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/ckpca',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)