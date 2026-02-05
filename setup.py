from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Minimal core dependencies - always installed
core_requirements = [
    # Core utilities
    "python-dotenv>=1.0.0",
    "requests>=2.31.0",
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
    "tenacity>=8.2.0",
    
    # Essential for agents
    "numpy>=1.24.0",
    
    # LLM Providers (must have for agents to work)
    "anthropic>=0.18.0",
    "openai>=1.0.0",
    "groq>=0.4.0",
    "together>=0.2.0",
    "ollama>=0.1.0",
]

setup(
    name="kite-agent",
    version="0.1.0",
    author="Thien Nguyen",
    description="Production-Ready Agentic AI Framework with Enterprise Safety",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thienzz/Kite",
    project_urls={
        "Bug Tracker": "https://github.com/thienzz/Kite/issues",
        "Documentation": "https://github.com/thienzz/Kite/tree/main/docs",
        "Source Code": "https://github.com/thienzz/Kite",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        # LLM Providers
        "llm": [
            "anthropic>=0.18.0",
            "openai>=1.0.0",
            "groq>=0.4.0",
            "together>=0.2.0",
            "ollama>=0.1.0",
        ],
        
        # Vector memory and embeddings
        "memory": [
            "faiss-cpu>=1.7.4",
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "fastembed>=0.1.0",
            "rank-bm25>=0.2.2",
            "cohere>=5.0.0",
        ],
        
        # Database connectors
        "database": [
            "psycopg2-binary>=2.9.0",
            "redis>=5.0.0",
            "mysql-connector-python>=8.0.0",
            "motor>=3.3.0",
        ],
        
        # Monitoring and API
        "monitoring": [
            "prometheus-client>=0.19.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "python-multipart>=0.0.6",
        ],
        
        # Development tools
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        
        # Install everything (for full featured experience)
        "all": [
            # LLM
            "anthropic>=0.18.0",
            "openai>=1.0.0",
            "groq>=0.4.0",
            "together>=0.2.0",
            "ollama>=0.1.0",
            # Memory
            "faiss-cpu>=1.7.4",
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "fastembed>=0.1.0",
            "rank-bm25>=0.2.2",
            "cohere>=5.0.0",
            # Database
            "psycopg2-binary>=2.9.0",
            "redis>=5.0.0",
            "mysql-connector-python>=8.0.0",
            "motor>=3.3.0",
            # Monitoring
            "prometheus-client>=0.19.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "python-multipart>=0.0.6",
        ],
    },
    keywords="ai agents llm production-ready safety circuit-breaker agentic-ai",
)