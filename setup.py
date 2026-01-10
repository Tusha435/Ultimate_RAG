"""Setup script for Ultimate RAG."""

from setuptools import setup, find_packages

setup(
    name="ultimate-rag",
    version="1.0.0",
    description="Multi-modal Retrieval-Augmented Generation for Technical Documents",
    author="Ultimate RAG Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pymupdf>=1.23.0",
        "pdfplumber>=0.10.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
        "cli": [
            "typer>=0.9.0",
            "rich>=13.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ],
        "full": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "typer>=0.9.0",
            "rich>=13.0.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "streamlit>=1.28.0",
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "sympy>=1.12",
        ]
    },
    entry_points={
        "console_scripts": [
            "ultimate-rag=ultimate_rag.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
)
