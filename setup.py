from setuptools import setup, find_packages

setup(
    name='intv',
    version='0.2.0',
    description='Web and app with OCR/RAG data retrieval, voice recognition & transcription (fast-whisper with VAR and diarization) for generating CPS interview documentation',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",    
    author='David Anderson',
    url="https://github.com/ErebusNyx/INTV",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'PyPDF2',
        'python-docx',
        'requests',
        'pdf2image',
        'pytesseract',
        'torch',  # For CUDA detection
        'psutil',
        'fastapi',
        'uvicorn',
    ],
    entry_points={
        'console_scripts': [
            'intv=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Utilities",
        "Framework :: FastAPI",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        '': ['*.json', '*.yaml', '*.yml', '*.md'],
    },
)
