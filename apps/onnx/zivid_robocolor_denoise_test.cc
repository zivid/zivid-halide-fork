#include "zivid_robocolor_denoise_model.h"

#include <Halide.h>
#include <HalideRuntimeOpenCL.h>

#include <iostream>
#include <chrono>

int main()
{
    // std::cout << "Selecting GPU device...\n";
    // halide_set_gpu_device(1);

    std::cout << "Running zivid robocolor denoise model test...\n";
    constexpr size_t image_size = 2848;
    Halide::Runtime::Buffer<float, 4> input(1, 3, image_size, image_size);
    Halide::Runtime::Buffer<float, 4> output(1, 3, image_size, image_size);

    std::mt19937 rnd(123);
    input.for_each_value([&](float &v) {
        v = rnd();
    });
    input.set_host_dirty();

    // std::cout << "Copying input to device...\n";
    // const auto *const device_interface = halide_opencl_device_interface();
    // input.copy_to_device(device_interface);

    std::cout << "Input shape: " << input.dim(0).extent() << "x"
              << input.dim(1).extent() << "x" << input.dim(2).extent() << "x"
              << input.dim(3).extent() << "\n";

    std::cout << "Running Warmup ...\n";
    Halide::Runtime::Buffer<float, 4> warmup_output(1, 3, image_size, image_size);
    zivid_robocolor_denoise_model(input, warmup_output);
    warmup_output.device_sync();
    std::cout << "Warmup completed.\n";

    std::cout << "Running inference...\n";
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    zivid_robocolor_denoise_model(input, output);
    output.device_sync();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const auto inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference completed in " << inferenceTime.count() << " ms\n";
    output.copy_to_host();

    for (int b = 0; b < 1; ++b) {
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 502; ++i) {
                for (int j = 0; j < 502; ++j) {
                    if (input(b, c, i, j) == output(b, c, i, j)) {
                        std::cerr << "Unexpected value for inputs at (" << b << ", " << c << ", " << i << "," << j << ") \n";
                        return -1;
                    }
                }
            }
        }
    }
    std::cout << "Success!\n";
    return 0;
}