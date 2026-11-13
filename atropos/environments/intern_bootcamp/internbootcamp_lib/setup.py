import setuptools

setuptools.setup(
    name="internbootcamp",
    version="0.1.0",
    url="https://github.com/InternLM/InternBootcamp/tree/main",
    packages=setuptools.find_packages(include=['internbootcamp',]),
    install_requires=[
        "distance",
        "matplotlib",
        "datasets",
        "jsonlines",
        "fire",
        "Faker",
        "python-sat",
        "sympy",
        "openai",
        "openpyxl",
        "transformers",
        "langdetect",
        "pympler",
        "shortuuid"
    ],

    package_data={
        
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)