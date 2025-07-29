#include "zivid_robocolor_denoise_model.h"

#include <Halide.h>

#include <iostream>

int main()
{
    std::cout << "Running zivid robocolor denoise model test...\n";
    Halide::Runtime::Buffer<float, 4> input(1, 3, 502, 502);
    Halide::Runtime::Buffer<float, 4> output(1, 3, 492, 492);

    std::mt19937 rnd(123);
    input.for_each_value([&](float &v) {
        v = rnd();
    });

    zivid_robocolor_denoise_model(input, output);
    
    for (int b = 0; b < 1; ++b) {
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 502; ++i) {
                for (int j = 0; j < 502; ++j) {
                    if (input(b, c, i, j) != output(b, c, i, j)) {
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