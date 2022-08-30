INC_DIR=-I./include -I/usr/include -I./vkFFT
BIN_DIR=./bin
LIB_DIR=-L/usr/lib/x86_64-linux-gnu/

.PHONY : all test clean a b c d

WMMA_INC=/global/home/lulu/zsx_home/zsx-home/software/rocWMMA/include
ROC_INC=/opt/rocm-5.2.0/hip/include
ROCFFT_INC=/global/home/lulu/zsx_home/zsx-home/software/rocm-4.5.0/rocfft/include

ROC_LIB=/opt/rocm-5.2.0/lib
WMMA_LIB=/global/home/lulu/zsx_home/zsx-home/software/rocWMMA/lib
ROCFFT_LIB=/global/home/lulu/zsx_home/zsx-home/software/rocm-4.5.0/rocfft/lib

INC_DIR+=-I${WMMA_INC} -I${ROC_INC} -I${ROCFFT_INC}
LIB_DIR+=-L${WMMA_LIB} -L${ROC_LIB} -L${ROCFFT_LIB}

SRC_1D_T=./source/mcfft_1d_device.cpp ./source/mcfft_1d_utils.cpp ./source/vkfft_utils.cpp ./benchmark/test_1d.cpp
SRC_1D_A=./source/mcfft_1d_device.cpp ./source/mcfft_1d_utils.cpp ./source/vkfft_utils.cpp ./benchmark/accuracy_1d.cpp

SRC_2D_T=./source/mcfft_2d_device.cpp ./source/mcfft_2d_utils.cpp ./source/vkfft_utils.cpp ./benchmark/test_2d.cpp
SRC_2D_A=./source/mcfft_2d_device.cpp ./source/mcfft_2d_utils.cpp ./source/vkfft_utils.cpp ./benchmark/accuracy_2d.cpp

SRC_3D_T=./source/mcfft_3d_device.cpp ./source/mcfft_3d_utils.cpp ./source/vkfft_utils.cpp ./benchmark/test_3d.cpp
SRC_3D_A=./source/mcfft_3d_device.cpp ./source/mcfft_3d_utils.cpp ./source/vkfft_utils.cpp ./benchmark/accuracy_3d.cpp

BIN_TARGETS=./bin/test_1d ./bin/accuracy_1d ./bin/test_2d ./bin/accuracy_2d ./bin/test_3d ./bin/accuracy_3d

ALL:${BIN_TARGETS}

CC=/opt/rocm-5.2.0/bin/hipcc
CFLAGS= -g -w -std=c++14 
LIB_USE= -lfftw3 -lrocfft

./bin/test_1d:${SRC_1D_T}
	$(call Mkdir, ${BIN_DIR})
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} ${LIB_USE} ${SRC_1D_T} -o $@

./bin/accuracy_1d:${SRC_1D_A}
	$(call Mkdir, ${BIN_DIR})
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} ${LIB_USE} ${SRC_1D_A} -o $@

./bin/test_2d:${SRC_2D_T}
	$(call Mkdir, ${BIN_DIR})
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} ${LIB_USE} ${SRC_2D_T} -o $@

./bin/accuracy_2d:${SRC_2D_A}
	$(call Mkdir, ${BIN_DIR})
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} ${LIB_USE} ${SRC_2D_A} -o $@

./bin/test_3d:${SRC_3D_T}
	$(call Mkdir, ${BIN_DIR})
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} ${LIB_USE} ${SRC_3D_T} -o $@

./bin/accuracy_3d:${SRC_3D_A}
	$(call Mkdir, ${BIN_DIR})
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} ${LIB_USE} ${SRC_3D_A} -o $@

define Mkdir
	$(shell if [ ! -d $(1) ]; then mkdir $(1); fi)
endef


test:
#	$(call Mkdir, ${OBJ_DIR})
	@echo $(BIN_TARGETS)
#	echo $(OBJ)
#	echo $(INC_DIR)

a:
	./bin/test_1d -n 256 -b 262144

b:
	./bin/accuracy -n 131072 -b 1

c:
	LD_LIBRARY_PATH=/home/lulu/zsx-home/software/rocWMMA/lib:/home/lulu/zsx-home/software/rocm-4.5.0/rocfft/lib:$$LD_LIBRARY_PATH

d:
	././bin/accuracy -n 256 -b 512
