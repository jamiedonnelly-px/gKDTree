include .env

build:
	@mkdir -p build && cd build && CUDACXX=/usr/local/cuda-12/bin/nvcc cmake -DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native .. && make -j4

clean:
	@rm -rf build
	@find . -name *.so | xargs -I {} rm -f {}
	@find . -name __pycache__ | xargs -I {} rm -rf {}
	@find . -name *.pyd | xargs -I {} rm -f {}
	@find . -name *.egg* | xargs -I {} rm -rf {}