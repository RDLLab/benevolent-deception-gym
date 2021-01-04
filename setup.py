from setuptools import setup, find_packages

extras = {
    'docs': [
        'sphinx>=3.0',
        'sphinx-rtd-theme>=0.4'
    ],
    'test': [
        'pytest>=5.4'
    ]
}

extras['all'] = [item for group in extras.values() for item in group]

setup(
    name='bdgym',
    version='0.0.1',
    # url="https://XXX.readthedocs.io",
    description=(
        "Multiagent environments that incoporate benevolent deception "
        "and implement the Open AI Gym interface"
    ),
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author="Jonathon Schwartz",
    author_email="Jonathon.Schwartz@anu.edu.au",
    license="MIT",
    packages=[
        package for package in find_packages()
        if package.startswith('bdgym')
    ],
    install_requires=[
        'gym>=0.17',
    ],
    extras_require=extras,
    python_requires='>=3.7',
    project_urls={
        # 'Documentation': "https://networkattacksimulator.readthedocs.io",
        'Source': "https://github.com/Jjschwartz/benevolent-deception-gym",
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    zip_safe=False
)
