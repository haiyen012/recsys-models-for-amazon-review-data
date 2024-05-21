SHELL = /bin/bash
PYTHON := python3
VENV_NAME = dlrm_env
FEATURE_FOLDER = feature_manager
MODEL_FOLDER = models

# Environment
venv:
	conda create --name ${VENV_NAME} python=3.9 && \
	conda init bash && \
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate ${VENV_NAME} && \
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install -e .[dev] && \
	pre-commit install


# Style
style:
	black ./${FEATURE_FOLDER}/ ./${MODEL_FOLDER}/
	flake8 ./${FEATURE_FOLDER}/ ./${MODEL_FOLDER}/
	${PYTHON} -m isort -rc ./${FEATURE_FOLDER}/ ./${MODEL_FOLDER}/


# Test
test:
	${PYTHON} -m flake8 ./${FEATURE_FOLDER}/ ./${MODEL_FOLDER}/
	${PYTHON} -m mypy ./${FEATURE_FOLDER}/ ./${MODEL_FOLDER}/
	# pytest -s --durations=0 --disable-warnings ./${MODEL_FOLDER}/
