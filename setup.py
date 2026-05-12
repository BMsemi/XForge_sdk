from setuptools import setup, find_packages

setup(
    name="XForge",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch",
        "Pillow",
        "ultralytics",
        "opencv-python",
    ],
    author="BMLabs",
    description="An SDK for Neuromorphic Hardware Simulation and Object Detection",
)
