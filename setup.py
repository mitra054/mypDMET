import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='pDMET',
        version="1.3.0",
        description='Density matrix embedding theory for periodic systems',
        author='Hung Pham',
        author_email='pqh3.14@gmail.com',
        url="https://github.com/hungpham2017/pDMET.git",
        license='Apache 2.0',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.15.2',
            'scipy>=1.1.0',
            'pybind11>=2.2.3',            
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],


        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.6.3',
        ],
        zip_safe=True,
    )
