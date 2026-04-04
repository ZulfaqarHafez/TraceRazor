from setuptools import setup, find_packages

setup(
    name="tracerazor-langgraph",
    version="0.1.0",
    description="TraceRazor LangGraph callback adapter",
    author="Zulfaqar Hafez",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langchain-core>=0.2",
    ],
    extras_require={
        "langgraph": ["langgraph>=0.2"],
        "openai": ["openai>=1.0"],
    },
)
