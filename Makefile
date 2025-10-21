include .env

clean:
	@rm -rf build
	@find . -name *.so | xargs -I {} rm -f {}
	@find . -name __pycache__ | xargs -I {} rm -rf {}
	@find . -name *.pyd | xargs -I {} rm -f {}
	@find . -name *.egg* | xargs -I {} rm -rf {}