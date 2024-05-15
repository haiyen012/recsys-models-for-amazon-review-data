## Table of Contents

- [Installation](#installation)
- [About the Data](#data)
- [Running the Project](#run)

## Installation
After cloning this repository, follow these steps to set up the environment:

```
make venv
conda activate dlrm_env
```

## About the Data
- This repository is dedicated to analyzing the Digital Music section of Amazon's review dataset.
- To get started, you need to obtain the Amazon review data and download two specific files into the 'data' directory: Digital_Music.json (approximately 755.2MB) and meta_Digital_Music.json (approximately 66.5MB).

## Running the Project
To process features, run:

```
sh generating_features.sh
```

To build and train the model, run:

```
sh build_model.sh
```
If you are on a Windows system, run the following commands instead:
```
python main.py generating-features
python main.py training-model
```
