INC_DIR=
BIN_DIR=
SRC_DIR=./source
OBJ_DIR=./obj
LIB_DIR=
OoO=%i

#伪目标
.PHONY : test clean debug


WMMA_INC=/global/home/lulu/zsx_home/zsx-home/software/rocWMMA/include
#WMMA_INC=/home/lulu/zsx-home/software/rocWMMA/include
ROC_INC=/opt/rocm-4.5.0/hip/include

ROC_LIB=/opt/rocm-4.5.0/lib
#WMMA_LIB=/home/lulu/zsx-home/software/rocWMMA/lib
WMMA_LIB=/global/home/lulu/zsx_home/zsx-home/software/rocWMMA/lib

INC_DIR+=-I${WMMA_INC} -I${ROC_INC}
LIB_DIR+=-L${WMMA_LIB} -L${ROC_LIB}
#要生成的目标文件路径和名称
TARGET=prog
SOURCE=t1.cpp

#用到的编译器和编译选项
CC=hipcc
CFLAGS= -g -w -std=c++14 -DENABLE_S2_GEOMETRY=0

# $@为编译目标名称,这里就是${BIN_TARGET}，也就是prog，这个命令实际上类似g++ *.o -o prog
${TARGET}:${SOURCE}
	${CC} ${CFLAGS} ${INC_DIR} ${LIB_DIR} $<  -o $@

#用于查看变量的值
test:
#	$(call Mkdir, ${OBJ_DIR})
#	echo $(SRC)
#	echo $(OBJ)
#	echo $(INC_DIR)

debug:
	./prog
