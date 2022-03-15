#include "HalideBuffer.h"
#include "reprod.h"
#include <iostream>

extern "C"
{
struct cl_context;
struct cl_command_queue;
// NOLINTNEXTLINE(readability-identifier-naming)
extern int halide_acquire_cl_context(void *, cl_context *ctx, cl_command_queue *queue, bool create = true);
// NOLINTNEXTLINE(readability-identifier-naming)
extern int halide_release_cl_context(void *);
// NOLINTNEXTLINE(readability-identifier-naming)
extern void halide_opencl_cleanup();
extern void halide_cache_cleanup();
extern void halide_thread_pool_cleanup();
}

int main() {
    {
        Halide::Runtime::Buffer<double> out(8);
        consumer(out);
        out.copy_to_host();
        for (double i : out) {
            std::cerr << i << ' ';
        }
        std::cerr << std::endl;
    }
    consumer_buffer_cleanup();

    {
        Halide::Runtime::Buffer<double> out(8);
        consumer(out);
        out.copy_to_host();
        for (double i : out) {
            std::cerr << i << ' ';
        }
        std::cerr << std::endl;
    }

    halide_thread_pool_cleanup();
    halide_cache_cleanup();
    halide_opencl_cleanup();
}
