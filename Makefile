WARNINGS=-Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wcast-align -Xcompiler -Wpointer-arith -Xcompiler -Wcast-qual -Xcompiler -Wstrict-aliasing=2 -Xcompiler -Wstrict-overflow=5

release:
	nvcc ${WARNINGS} tensor.cu -L${CUTENSOR_ROOT}/lib/12/ -I${CUTENSOR_ROOT}/include -std=c++20 -O3 -lcutensor -o tensor

debug:
	nvcc ${WARNINGS} tensor.cu -L${CUTENSOR_ROOT}/lib/12/ -I${CUTENSOR_ROOT}/include -DDEBUG -std=c++20 -lcutensor -o tensor
