include .env

.PHONY: install-package

env: 
	@conda create -y -n ${CONDA_ENV_NAME} python=${PYTHON_VER}

rmenv:
	@conda remove --all -y -n ${CONDA_ENV_NAME}

install-package:
	@pip install --force-reinstall -v . 

clean:
	@rm -rf build
	@find . -name *.so | xargs -I {} rm -f {}
	@find . -name __pycache__ | xargs -I {} rm -rf {}
	@find . -name *.pyd | xargs -I {} rm -f {}
	@find . -name *.egg* | xargs -I {} rm -rf {}