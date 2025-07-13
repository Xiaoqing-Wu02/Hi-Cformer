from setuptools import setup, find_packages

setup(
    name='hicformer',
    version='0.0.1',
    description='Hi-Cformer model for single-cell Hi-C analysis',
    author='Xiaoqing Wu',
    author_email="xq-wu24@mails.tsinghua.edu.cn",
    url="https://github.com/Xiaoqing-Wu02/Hi-Cformer",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "h5py",
        "hdf5storage",
        "mat73",
        "scanpy",
        "anndata",
        "episcanpy",
        "statsmodels",
        "umap-learn",
        "torch>=2.0.0",    
        "torchvision",      # Version/cuda handled by torch
        "torchaudio",       # Version/cuda handled by torch
        "timm",
        "tqdm"
    ]
,
    python_requires='>=3.9',
)
