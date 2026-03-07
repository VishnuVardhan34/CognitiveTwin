from setuptools import setup, find_packages

setup(
    name="cognitivetwin",
    version="1.0.0",
    description="Real-Time Digital Twin of Human Cognitive Load via EEG + Eye Tracking + HRV Fusion",
    author="Sumanth Kotikalapudi, Sai Charna Kukkala, Sumeeth Kumar, Vishnu Nutalapati",
    packages=find_packages(exclude=["tests*", "notebooks*", "frontend*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "mne>=1.4.0",
        "PyWavelets>=1.4.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "websockets>=11.0",
        "protobuf>=4.23.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "brainflow>=5.8.0",
        "pytest>=7.3.0",
    ],
    entry_points={
        "console_scripts": [
            "cognitivetwin-train=training.train_multimodal:main",
            "cognitivetwin-server=backend.websocket_server:main",
        ],
    },
)
