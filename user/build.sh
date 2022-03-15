#!/bin/bash

set -e

REPO_ROOT=/home/nishanth/source/Halide

rm *.o *.so *.ll *.txt *.c *.h || true

. $REPO_ROOT/venv/bin/activate
python $REPO_ROOT/user/reprod.py

HALIDE_HEADERS=$REPO_ROOT/cmake-build-debug/include

g++ -I$HALIDE_HEADERS reprod.c -c -o reprod.c.o -fPIC
g++ -o libreprod.so -shared reprod.c.o halide_runtime.o
g++ -I$HALIDE_HEADERS -o main main.cpp -L. -Wl,--rpath=. -lreprod
echo C CodeGen
./main || exit $?
exit 0

g++ -o libreprod.so -shared reprod.o halide_runtime.o
g++ -I$HALIDE_HEADERS -o main main.cpp -L. -Wl,--rpath=. -lreprod
echo LLVM CodeGen
./main || exit $?

