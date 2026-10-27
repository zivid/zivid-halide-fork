#include "Halide.h"

#include <map>

using namespace Halide;

// Checks that halide_set_cuda_device_allocator and
// halide_set_opencl_device_allocator serve the runtime's buffer allocations
// and frees, bypass the allocation cache, and that nullptr restores the
// default path.

struct HalideDeviceAllocator {
    void *allocate = nullptr;
    void *free = nullptr;
};

static void *(*get_symbol)(void *, const char *) = nullptr;
static int (*cuMemAlloc)(uint64_t *, size_t) = nullptr;
static int (*cuMemFree)(uint64_t) = nullptr;
static void *(*clCreateBuffer)(void *, uint64_t, size_t, void *, int *) = nullptr;
static int (*clReleaseMemObject)(void *) = nullptr;

static std::map<uint64_t, size_t> live;
static int allocations = 0;
static int frees = 0;

static int cuda_allocate(void *, void *, void *, size_t size, uint64_t *out) {
    int err = cuMemAlloc(out, size);
    if (err) {
        return halide_error_code_device_malloc_failed;
    }
    live[*out] = size;
    allocations++;
    return halide_error_code_success;
}

static int cuda_free(void *, void *, void *, uint64_t ptr, size_t size) {
    if (live.count(ptr) == 0 || live[ptr] != size) {
        printf("Freed %p with size %d that the allocator did not allocate\n", (void *)ptr, (int)size);
        exit(1);
    }
    live.erase(ptr);
    frees++;
    return cuMemFree(ptr) ? halide_error_code_device_free_failed : halide_error_code_success;
}

static int opencl_allocate(void *, void *context, void *, size_t size, void **out) {
    const uint64_t read_write = 1 << 0;
    int err = 0;
    *out = clCreateBuffer(context, read_write, size, nullptr, &err);
    if (err) {
        return halide_error_code_device_malloc_failed;
    }
    live[(uint64_t)*out] = size;
    allocations++;
    return halide_error_code_success;
}

static int opencl_free(void *, void *, void *, void *mem, size_t size) {
    if (live.count((uint64_t)mem) == 0 || live[(uint64_t)mem] != size) {
        printf("Freed %p with size %d that the allocator did not allocate\n", mem, (int)size);
        exit(1);
    }
    live.erase((uint64_t)mem);
    frees++;
    return clReleaseMemObject(mem) ? halide_error_code_device_free_failed : halide_error_code_success;
}

static void *find_runtime_symbol(const Target &target, const char *name) {
    for (Internal::JITModule &m : Internal::JITSharedRuntime::get(nullptr, target, false)) {
        auto symbol = m.find_symbol_by_name(name);
        if (symbol.address != nullptr) {
            return symbol.address;
        }
    }
    return nullptr;
}

static void check(int result, const char *what) {
    if (result != 0) {
        printf("%s failed: %d\n", what, result);
        exit(1);
    }
}

static int run_pipeline(int width) {
    Func f, g;
    Var x, xi;
    f(x) = x * 2;
    g(x) = f(x) + f(x + 1);
    f.compute_root().gpu_tile(x, xi, 16);
    g.gpu_tile(x, xi, 16);
    Buffer<int> out = g.realize({width});
    out.copy_to_host();
    for (int i = 0; i < out.width(); i++) {
        if (out(i) != i * 4 + 2) {
            printf("out(%d) = %d instead of %d\n", i, out(i), i * 4 + 2);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    DeviceAPI api;
    const char *set_allocator_name;
    if (target.has_feature(Target::CUDA)) {
        api = DeviceAPI::CUDA;
        set_allocator_name = "halide_set_cuda_device_allocator";
    } else if (target.has_feature(Target::OpenCL)) {
        api = DeviceAPI::OpenCL;
        set_allocator_name = "halide_set_opencl_device_allocator";
    } else {
        printf("[SKIP] Neither CUDA nor OpenCL enabled.\n");
        return 0;
    }

    // Load the runtime module.
    evaluate_may_gpu<float>(Expr(0.f));

    const halide_device_interface_t *iface = get_device_interface_for_device_api(api, target);
    auto set_allocator = (const HalideDeviceAllocator *(*)(const HalideDeviceAllocator *))
        find_runtime_symbol(target, set_allocator_name);
    get_symbol = (void *(*)(void *, const char *))
        find_runtime_symbol(target, api == DeviceAPI::CUDA ? "halide_cuda_get_symbol" : "halide_opencl_get_symbol");
    if (iface == nullptr || set_allocator == nullptr || get_symbol == nullptr) {
        printf("Failed to get the runtime\n");
        return 1;
    }
    HalideDeviceAllocator allocator;
    if (api == DeviceAPI::CUDA) {
        cuMemAlloc = (decltype(cuMemAlloc))get_symbol(nullptr, "cuMemAlloc_v2");
        cuMemFree = (decltype(cuMemFree))get_symbol(nullptr, "cuMemFree_v2");
        allocator = {(void *)cuda_allocate, (void *)cuda_free};
    } else {
        clCreateBuffer = (decltype(clCreateBuffer))get_symbol(nullptr, "clCreateBuffer");
        clReleaseMemObject = (decltype(clReleaseMemObject))get_symbol(nullptr, "clReleaseMemObject");
        allocator = {(void *)opencl_allocate, (void *)opencl_free};
    }

    Internal::JITSharedRuntime::reuse_device_allocations(true);

    if (set_allocator(&allocator) != nullptr) {
        printf("Expected no allocator to be set\n");
        return 1;
    }

    printf("Test that allocations and frees go through the allocator, with the buffer size.\n");
    {
        Buffer<float> a(1000);
        check(a.device_malloc(iface), "device_malloc");
        if (allocations != 1 || live.begin()->second != 1000 * sizeof(float)) {
            printf("Expected one allocation of the buffer size, got %d\n", allocations);
            return 1;
        }
        check(a.device_free(), "device_free");
        if (frees != 1 || !live.empty()) {
            printf("Expected the allocation to be freed through the allocator\n");
            return 1;
        }
    }

    printf("Test that the allocation cache is bypassed.\n");
    {
        for (int i = 0; i < 3; i++) {
            Buffer<float> a(1000);
            check(a.device_malloc(iface), "device_malloc");
        }
        if (allocations != 4 || frees != 4) {
            printf("Expected every allocation to reach the allocator, got %d allocations and %d frees\n", allocations, frees);
            return 1;
        }
    }

    printf("Test that pipeline intermediates go through the allocator.\n");
    {
        if (run_pipeline(1000)) {
            return 1;
        }
        if (allocations < 6 || allocations != frees || !live.empty()) {
            printf("Expected the intermediate and output to use the allocator, got %d allocations and %d frees\n", allocations, frees);
            return 1;
        }
    }

    printf("Test that nullptr restores the default allocation.\n");
    {
        if (set_allocator(nullptr) != &allocator) {
            printf("Expected the previous allocator to be returned\n");
            return 1;
        }
        int before = allocations;
        if (run_pipeline(1000)) {
            return 1;
        }
        if (allocations != before) {
            printf("Expected no allocations through the allocator once it is unset\n");
            return 1;
        }
    }

    iface->device_release(nullptr, iface);
    printf("Success!\n");
    return 0;
}
