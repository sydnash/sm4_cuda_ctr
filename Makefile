.PHONY: all clean
CXX ?= nvcc
FALGS ?= -g -Wall -O2
GMSSL_ROOT ?=

TARGET = sm4_ctr

SOURCES= main.cc sm4_cuda.h sm4_cuda.cu

all: ${TARGET}

ifneq (,$(GMSSL_ROOT))
  GMSSL_EXISTS = $(wildcard $(GMSSL_ROOT))
  ifneq (,$(GMSSL_EXISTS))
    CXXFLAGS += -DGMSSL_TEST
    CXXFLAGS += -I$(GMSSL_ROOT)/include
    LDFLAGS += -L$(GMSSL_ROOT)/lib -lgmssl
    LDLIBS += -Wl,-rpath,$(GMSSL_ROOT)/lib
  else
    $(warning Warning: GMSSL_ROOT path '$(GMSSL_ROOT)' does not exist. GMSSL support will be disabled.)
  endif
endif

CXXFLAGS += $(FLAGS)

${TARGET}: ${SOURCES}
	${CXX} ${CXXFLAGS} -std=gnu++11 -o $@ ${SOURCES}

clean:
	rm -f ${TARGET}