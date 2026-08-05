from setuptools import setup, find_packages

setup(
    name="promptda",
    version="1.0",
    packages=find_packages(include=["promptda", "promptda.*"]),
    author="Haotong Lin",
    description="Prompt Depth Anything",
    url="https://github.com/DepthAnything/PromptDA",
)
