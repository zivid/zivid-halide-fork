#ifndef HALIDE_HALIDERUNTIMECUDA_H
#define HALIDE_HALIDERUNTIMECUDA_H

// Don't include HalideRuntime.h if the contents of it were already pasted into a generated header above this one
#ifndef HALIDE_HALIDERUNTIME_H

#include "HalideRuntime.h"

#endif

#ifdef __cplusplus
extern "C" {
#endif

/** \file
 *  Routines specific to the Halide Cuda runtime.
 */

#define HALIDE_RUNTIME_CUDA

extern const struct halide_device_interface_t *halide_cuda_device_interface();

/** These are forward declared here to allow clients to override the
 *  Halide Cuda runtime. Do not call them. */
// @{
extern int halide_cuda_initialize_kernels(void *user_context, void **state_ptr,
                                          const char *src, int size);
extern int halide_cuda_run(void *user_context,
                           void *state_ptr,
                           const char *entry_name,
                           int blocksX, int blocksY, int blocksZ,
                           int threadsX, int threadsY, int threadsZ,
                           int shared_mem_bytes,
                           size_t arg_sizes[],
                           void *args[],
                           int8_t arg_is_buffer[]);
extern void halide_cuda_finalize_kernels(void *user_context, void *state_ptr);
// @}

/** Set the underlying cuda device poiner for a buffer. The device
 * pointer should be allocated using cuMemAlloc or similar and must
 * have an extent large enough to cover that specified by the
 * halide_buffer_t extent fields. The dev field of the halide_buffer_t
 * must be NULL when this routine is called. This call can fail due to
 * being passed an invalid device pointer. The device and host dirty
 * bits are left unmodified. */
extern int halide_cuda_wrap_device_ptr(void *user_context, struct halide_buffer_t *buf, uint64_t device_ptr);

/** Disconnect this halide_buffer_t from the device pointer it was
 * previously wrapped around. Should only be called for a
 * halide_buffer_t that halide_cuda_wrap_device_ptr was previously
 * called on. The device field of the halide_buffer_t will be NULL on
 * return.
 */
extern int halide_cuda_detach_device_ptr(void *user_context, struct halide_buffer_t *buf);

/** Return the underlying device pointer for a halide_buffer_t. This buffer
 *  must be valid on a Cuda device, or not have any associated device
 *  memory. If there is no device memory (dev field is NULL), this
 *  returns 0.
 */
extern uintptr_t halide_cuda_get_device_ptr(void *user_context, struct halide_buffer_t *buf);

/** Release any currently-unused device allocations back to the cuda
 * driver. See halide_reuse_device_allocations. */
extern int halide_cuda_release_unused_device_allocations(void *user_context);

// These typedefs treat both a CUcontext and a CUstream as a void *,
// to avoid dependencies on cuda headers.
typedef int (*halide_cuda_acquire_context_t)(void *,   // user_context
                                             void **,  // cuda context out parameter
                                             bool);    // should create a context if none exist
typedef int (*halide_cuda_release_context_t)(void * /* user_context */);
typedef int (*halide_cuda_get_stream_t)(void *,    // user_context
                                        void *,    // context
                                        void **);  // stream out parameter

/** Set custom methods to acquire and release cuda contexts and streams */
// @{
extern halide_cuda_acquire_context_t halide_set_cuda_acquire_context(halide_cuda_acquire_context_t handler);
extern halide_cuda_release_context_t halide_set_cuda_release_context(halide_cuda_release_context_t handler);
extern halide_cuda_get_stream_t halide_set_cuda_get_stream(halide_cuda_get_stream_t handler);
// @}

/** Device memory for halide_cuda_device_malloc and halide_cuda_device_free.
 * Both are called with the cuda context current, and with the stream from
 * halide_cuda_get_stream. The memory may be released on that stream while
 * work queued on it still uses the memory, so free must order the release
 * after that work, for example with cuMemFreeAsync. size is the size of the
 * buffer, and is the same in free as in the allocate it pairs with. */
struct halide_cuda_device_allocator_t {
    int (*allocate)(void *user_context, void *context, void *stream, size_t size, uint64_t *device_ptr_out);
    int (*free)(void *user_context, void *context, void *stream, uint64_t device_ptr, size_t size);
};

/** Serve device allocations from allocator instead of cuMemAlloc, cuMemFree
 * and the allocation cache, or restore those with nullptr. Change it only
 * while no device memory allocated by the cuda runtime is alive, since
 * memory is freed through the allocator in effect when it is freed. Returns
 * the previous allocator. */
extern const struct halide_cuda_device_allocator_t *halide_set_cuda_device_allocator(const struct halide_cuda_device_allocator_t *allocator);

#ifdef __cplusplus
}  // End extern "C"
#endif

#endif  // HALIDE_HALIDERUNTIMECUDA_H
