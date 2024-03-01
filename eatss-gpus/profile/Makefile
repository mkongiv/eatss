NVCC = nvcc
CC = gcc
CFLAGS = -O3 -lm 
NVCCFLAGS = $(NVCC_FLAGS)

SRC = $(wildcard $(TARGET)*.cu)
OBJ = $(patsubst %.cu,%.o,$(SRC))

C_SRC = ../polybench-c-3.2/utilities/polybench.c 
C_OBJ = $(patsubst %.c,%.o,$(C_SRC))

TARGET = $(PROG_TARGET)
TYPE = $(PROG_TYPE)
CATEGORY = $(PROG_CATEGORY)

ifdef PROG_TYPE
INC =-I ../polybench-c-3.2/utilities -I ../polybench-c-3.2/$(CATEGORY)/$(TYPE)/$(TARGET)/
else
INC =-I ../polybench-c-3.2/utilities -I ../polybench-c-3.2/$(CATEGORY)/$(TARGET)/
endif


all: $(TARGET)

%.o: %.cu 
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(INC) -dc -o $@ $< 

%.o: %.c 
	$(CC) $(CFLAGS) $(INC) -c -o $@ $< 

polybench.o: $(C_SRC) 
	$(CC) $(CFLAGS) $(INC) -c -o $@ $< 

$(TARGET): $(C_OBJ) $(OBJ) 
	$(NVCC) $(CFLAGS) $(INC) -o $(TARGET) $^ -lcuda -lcudart_static

.PHONY: clean

clean:
	rm *.o $(TARGET)
