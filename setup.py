from setuptools import setup, find_packages

setup(
    name='rlxf',
    version='0.1.0',
    description='Framework to automatically build preference datasets for LLM alignment',
    author='Argilla Team',
    author_email='admin@argilla.io',
    url='https://github.com/argilla-io/rlxf',
    packages=find_packages(),
    install_requires=[
        # List your library's dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
    ],
)