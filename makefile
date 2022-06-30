INC_DIR=-I./include -I/usr/include -I./vkFFT
BIN_DIR=./bin
SRC_DIR=./source
LIB_DIR=-L/usr/lib/x86_64-linux-gnu/
EXC_DIR=./start

.PHONY : all test clean a b c d

WMMA_INC=/home/lulu/zsx-home/software/rocWMMA/include
ROC_INC=/opt/rocm-4.5.2/hip/include
ROCFFT_INC=/home/lulu/zsx-home/software/rocm-4.5.0/rocfft/include

ROC_LIB=/opt/rocm-4.5.2/lib
WMMA_LIB=/home/lulu/zsx-home/software/rocWMMA/lib
ROCFFT_LIB=/home/lulu/zsx-home/software/rocm-4.5.0/rocfft/lib

INC_DIR+=-I${WMMA_INC} -I${ROC_INC} -I${ROCFFT_INC}
LIB_DIR+=-L${WMMA_LIB} -L${ROC_LIB} -L${ROCFFT_LIB}

SRC=${wildcard ${SRC_DIR}/*.cpp}
EXC_SRC=${wildcard ${EXC_DIR}/*.cpp}
EXC=$(patsubst %.cpp, %, ${notdir ${wildcard ${EXC_DIR}/*.cpp}})


BIN_TARGETS+=$(foreach n,$(EXC),${BIN_DIR}/$(n))
all:${BIN_TARGETS}

CC=/opt/rocm-4.5.2/bin/hipcc
CFLAGS= -g -w -std=c++14 
LIB_USE= -lfftw3 -lrocfft


${BIN_TARGETS}:${SRC} ${EXC_SRC}
	$(call Mkdir, ${BIN_DIR})
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} ${LIB_USE} ${SRC} ${EXC_DIR}/$(notdir $@).cpp  -o $@



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
