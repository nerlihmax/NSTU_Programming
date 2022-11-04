EXECS=lab
COMPILER?=mpicxx

all: ${EXECS}

lab: main.cpp
	${COMPILER} -o lab -std=c++11 -g main.cpp

clean:
	rm ${EXECS}
