from setuptools import setup, find_packages

setup(
    name="face-comparison-api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "pydantic==2.5.0",
        "uvicorn==0.24.0",
        "requests==2.31.0",
        "numpy==1.24.3",
        "opencv-python-headless",
        "deepface",
        "scipy",
        "pandas",
        "mtcnn",
        "scikit-learn",
        "tensorflow",
    ],
)
