from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pricelib',
    version='1.0.0',
    packages=find_packages(),
    description='pricelib is an open-source financial derivatives pricing library written in Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='上海凌瓴信息科技有限公司',
    author_email='marx@galatech.com.cn',
    url='https://gitee.com/lltech/pricelib',
    install_requires=[
        'numpy>=1.24.1, <=1.26.4',
        'pandas>=2.0.3, <=2.2.2',
        'numba>=0.57.1, <=0.59.1',
        'scipy>=1.8.0, <=1.13.0',
        'importlib-metadata>=6.8.0, <=7.1.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pylint',
            'flake8',
        ],
        'plot': [
            'matplotlib>=3.5.3, <=3.7.5',
            'plotly>=5.16.1, <=5.22.0',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
