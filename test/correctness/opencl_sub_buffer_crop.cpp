#include "Halide.h"

using namespace Halide;

// A kernel argument that is a crop of a buffer whose cl_mem is itself a
// sub-buffer must still run. OpenCL does not allow a sub-buffer of a
// sub-buffer, so the runtime has to make the crop from the root buffer.

static void *find_runtime_symbol(const Target &target, const char *name) {
    for (Internal::JITModule &m : Internal::JITSharedRuntime::get(nullptr, target, false)) {
        auto symbol = m.find_symbol_by_name(name);
        if (symbol.address != nullptr) {
            return symbol.address;
        }
    }
    return nullptr;
}

struct ClRegion {
    size_t origin;
    size_t size;
};

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (!target.has_feature(Target::OpenCL)) {
        printf("[SKIP] OpenCL not enabled.\n");
        return 0;
    }

    // Load the runtime module.
    evaluate_may_gpu<float>(Expr(0.f));

    const halide_device_interface_t *iface = get_device_interface_for_device_api(DeviceAPI::OpenCL, target);
    auto get_symbol = (void *(*)(void *, const char *))find_runtime_symbol(target, "halide_opencl_get_symbol");
    auto acquire = (int (*)(void *, void **, void **, bool))find_runtime_symbol(target, "halide_acquire_cl_context");
    auto release = (int (*)(void *))find_runtime_symbol(target, "halide_release_cl_context");
    auto wrap = (int (*)(void *, halide_buffer_t *, uint64_t))find_runtime_symbol(target, "halide_opencl_wrap_cl_mem");
    auto detach = (int (*)(void *, halide_buffer_t *))find_runtime_symbol(target, "halide_opencl_detach_cl_mem");
    if (iface == nullptr || get_symbol == nullptr || acquire == nullptr || release == nullptr || wrap == nullptr || detach == nullptr) {
        printf("Failed to get the runtime\n");
        return 1;
    }
    auto clCreateBuffer = (void *(*)(void *, uint64_t, size_t, void *, int *))get_symbol(nullptr, "clCreateBuffer");
    auto clCreateSubBuffer = (void *(*)(void *, uint64_t, uint32_t, const void *, int *))get_symbol(nullptr, "clCreateSubBuffer");
    auto clReleaseMemObject = (int (*)(void *))get_symbol(nullptr, "clReleaseMemObject");

    void *context = nullptr;
    void *queue = nullptr;
    if (acquire(nullptr, &context, &queue, true) != 0) {
        printf("Failed to acquire the OpenCL context\n");
        return 1;
    }
    release(nullptr);

    const uint64_t read_write = 1 << 0;
    const uint32_t region_type = 0x1220;
    const int width = 4096;
    const int crop_min = 1024;
    const size_t piece_offset = 4096;
    int err = 0;
    void *root = clCreateBuffer(context, read_write, piece_offset + width * sizeof(float), nullptr, &err);
    if (err) {
        printf("clCreateBuffer failed: %d\n", err);
        return 1;
    }
    const ClRegion region{piece_offset, width * sizeof(float)};
    void *piece = clCreateSubBuffer(root, read_write, region_type, &region, &err);
    if (err) {
        printf("clCreateSubBuffer failed: %d\n", err);
        return 1;
    }

    {
        Buffer<float> input(width);
        input.for_each_element([&](int x) { input(x) = (float)x; });
        if (wrap(nullptr, input.raw_buffer(), (uint64_t)piece) != 0) {
            printf("Failed to wrap the sub-buffer\n");
            return 1;
        }
        input.set_host_dirty();
        input.copy_to_device(iface);

        Buffer<float> cropped = input.cropped(0, crop_min, width - crop_min);
        Func f;
        Var x, xi;
        f(x) = cropped(x) * 2.0f;
        f.gpu_tile(x, xi, 64);
        Buffer<float> output(width - crop_min);
        output.set_min(crop_min);
        f.realize(output);
        output.copy_to_host();
        for (int i = crop_min; i < width; i++) {
            if (output(i) != i * 2.0f) {
                printf("output(%d) = %f instead of %f\n", i, output(i), i * 2.0f);
                return 1;
            }
        }
        cropped = Buffer<float>();
        detach(nullptr, input.raw_buffer());
    }

    clReleaseMemObject(piece);
    clReleaseMemObject(root);
    printf("Success!\n");
    return 0;
}
