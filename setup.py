from setuptools import setup, find_packages

setup(
    name="livekit_uc",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "pydantic",
        "pyyaml",
        "requests",
        "tiktoken",
    ],
)
