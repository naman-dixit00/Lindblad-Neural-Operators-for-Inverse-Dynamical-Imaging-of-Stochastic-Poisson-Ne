from setuptools import setup, find_packages

setup(
    name="lno_ion_transport",
    version="0.1.0",
    author="Yuánlǐ AI Research Laboratory",
    description="Lindblad Neural Operators for Nanoscale Imaging of Dissipative Ion Transport in Neuronal Systems",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "h5py>=3.8.0",
        "scikit-learn>=1.2.0"
    ],
)
