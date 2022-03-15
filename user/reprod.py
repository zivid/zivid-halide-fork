from halide import *
from pathlib import Path
import numpy as np


def make_runtime(root_dir):
    module = Module("halide_runtime", runtime_target())
    module.compile(outputs={Output.object: str(root_dir / "halide_runtime.o")})


def find_gpu_target() -> Target:
    return runtime_target().with_feature(TargetFeature.NoRuntime)


def runtime_target() -> Target:
    host_target = get_host_target()
    return Target(host_target.os, host_target.arch, host_target.bits,
                  [TargetFeature.OpenCL, TargetFeature.CLDoubles, TargetFeature.Debug])


def main():
    lut_const = np.arange(128, dtype=float)
    lut_buffer = Buffer(lut_const)

    at_lut = Func("at_lut")
    i = Var()
    block, thread = Var(), Var()
    at_lut[i] = lut_buffer[i]
    at_lut.gpu_tile(i, block, thread, 1)
    # at_lut.split(i, block, thread, 1, TailStrategy.Auto)
    # at_lut.gpu_blocks(block)
    # at_lut.gpu_threads(thread)

    # consumer = Func("consumer")
    # x, y = Var(), Var()
    # consumer[x, y] = at_lut[(x + y) % lut_const.shape[0]] * (x * x + y)
    # xo, yo, xi, yi = Var(), Var(), Var(), Var()
    # consumer.gpu_tile(x, y, xo, yo, xi, yi, 8, 8)

    root_dir = Path.cwd()
    (root_dir / "include").mkdir(parents=True, exist_ok=True)
    at_lut.compile_to(outputs={
        Output.c_header: str(root_dir / "include" / "reprod.h"),
        Output.c_source: str(root_dir / "reprod.cc"),
        Output.llvm_assembly: str(root_dir / "reprod.ll"),
        Output.object: str(root_dir / "reprod.o"),
    }, arguments=[], fn_name="consumer", target=find_gpu_target())
    make_runtime(root_dir)


if __name__ == '__main__':
    main()
