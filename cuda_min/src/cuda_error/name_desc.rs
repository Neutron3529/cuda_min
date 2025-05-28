pub fn get_name_desc(code: std::ffi::c_int) -> (&'static str, &'static str) {
    match code {
        0 => ("CUDA_SUCCESS", "no error"),
        1 => ("CUDA_ERROR_INVALID_VALUE", "invalid argument"),
        2 => ("CUDA_ERROR_OUT_OF_MEMORY", "out of memory"),
        3 => ("CUDA_ERROR_NOT_INITIALIZED", "initialization error"),
        4 => ("CUDA_ERROR_DEINITIALIZED", "driver shutting down"),
        5 => (
            "CUDA_ERROR_PROFILER_DISABLED",
            "profiler disabled while using external profiling tool",
        ),
        6 => (
            "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
            "profiler not initialized: call cudaProfilerInitialize()",
        ),
        7 => (
            "CUDA_ERROR_PROFILER_ALREADY_STARTED",
            "profiler already started",
        ),
        8 => (
            "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
            "profiler already stopped",
        ),
        34 => ("CUDA_ERROR_STUB_LIBRARY", "CUDA driver is a stub library"),
        46 => (
            "CUDA_ERROR_DEVICE_UNAVAILABLE",
            "CUDA-capable device(s) is/are busy or unavailable",
        ),
        100 => ("CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"),
        101 => ("CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"),
        102 => (
            "CUDA_ERROR_DEVICE_NOT_LICENSED",
            "device doesn\'t have valid Grid license",
        ),
        200 => ("CUDA_ERROR_INVALID_IMAGE", "device kernel image is invalid"),
        201 => ("CUDA_ERROR_INVALID_CONTEXT", "invalid device context"),
        202 => (
            "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            "context already current",
        ),
        205 => ("CUDA_ERROR_MAP_FAILED", "mapping of buffer object failed"),
        206 => (
            "CUDA_ERROR_UNMAP_FAILED",
            "unmapping of buffer object failed",
        ),
        207 => ("CUDA_ERROR_ARRAY_IS_MAPPED", "array is mapped"),
        208 => ("CUDA_ERROR_ALREADY_MAPPED", "resource already mapped"),
        209 => (
            "CUDA_ERROR_NO_BINARY_FOR_GPU",
            "no kernel image is available for execution on the device",
        ),
        210 => ("CUDA_ERROR_ALREADY_ACQUIRED", "resource already acquired"),
        211 => ("CUDA_ERROR_NOT_MAPPED", "resource not mapped"),
        212 => (
            "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
            "resource not mapped as array",
        ),
        213 => (
            "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
            "resource not mapped as pointer",
        ),
        214 => (
            "CUDA_ERROR_ECC_UNCORRECTABLE",
            "uncorrectable ECC error encountered",
        ),
        215 => (
            "CUDA_ERROR_UNSUPPORTED_LIMIT",
            "limit is not supported on this architecture",
        ),
        216 => (
            "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
            "exclusive-thread device already in use by a different thread",
        ),
        217 => (
            "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
            "peer access is not supported between these two devices",
        ),
        218 => ("CUDA_ERROR_INVALID_PTX", "a PTX JIT compilation failed"),
        219 => (
            "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
            "invalid OpenGL or DirectX context",
        ),
        220 => (
            "CUDA_ERROR_NVLINK_UNCORRECTABLE",
            "uncorrectable NVLink error detected during the execution",
        ),
        221 => (
            "CUDA_ERROR_JIT_COMPILER_NOT_FOUND",
            "PTX JIT compiler library not found",
        ),
        222 => (
            "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
            "the provided PTX was compiled with an unsupported toolchain.",
        ),
        223 => (
            "CUDA_ERROR_JIT_COMPILATION_DISABLED",
            "PTX JIT compilation was disabled",
        ),
        224 => (
            "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY",
            "the provided execution affinity is not supported",
        ),
        225 => (
            "CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC",
            "the provided PTX contains unsupported call to cudaDeviceSynchronize",
        ),
        226 => (
            "CUDA_ERROR_CONTAINED",
            "Invalid access of peer GPU memory over nvlink or a hardware error",
        ),
        300 => (
            "CUDA_ERROR_INVALID_SOURCE",
            "device kernel image is invalid",
        ),
        301 => ("CUDA_ERROR_FILE_NOT_FOUND", "file not found"),
        302 => (
            "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
            "shared object symbol not found",
        ),
        303 => (
            "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
            "shared object initialization failed",
        ),
        304 => (
            "CUDA_ERROR_OPERATING_SYSTEM",
            "OS call failed or operation not supported on this OS",
        ),
        400 => ("CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"),
        401 => (
            "CUDA_ERROR_ILLEGAL_STATE",
            "the operation cannot be performed in the present state",
        ),
        402 => (
            "CUDA_ERROR_LOSSY_QUERY",
            "attempted introspection would be semantically lossy",
        ),
        500 => ("CUDA_ERROR_NOT_FOUND", "named symbol not found"),
        600 => ("CUDA_ERROR_NOT_READY", "device not ready"),
        700 => (
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "an illegal memory access was encountered",
        ),
        701 => (
            "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            "too many resources requested for launch",
        ),
        702 => (
            "CUDA_ERROR_LAUNCH_TIMEOUT",
            "the launch timed out and was terminated",
        ),
        703 => (
            "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
            "launch uses incompatible texturing mode",
        ),
        704 => (
            "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
            "peer access is already enabled",
        ),
        705 => (
            "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
            "peer access has not been enabled",
        ),
        708 => (
            "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            "cannot set while device is active in this process",
        ),
        709 => ("CUDA_ERROR_CONTEXT_IS_DESTROYED", "context is destroyed"),
        710 => ("CUDA_ERROR_ASSERT", "device-side assert triggered"),
        711 => (
            "CUDA_ERROR_TOO_MANY_PEERS",
            "peer mapping resources exhausted",
        ),
        712 => (
            "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
            "part or all of the requested memory range is already mapped",
        ),
        713 => (
            "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
            "pointer does not correspond to a registered memory region",
        ),
        714 => ("CUDA_ERROR_HARDWARE_STACK_ERROR", "hardware stack error"),
        715 => (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "an illegal instruction was encountered",
        ),
        716 => ("CUDA_ERROR_MISALIGNED_ADDRESS", "misaligned address"),
        717 => (
            "CUDA_ERROR_INVALID_ADDRESS_SPACE",
            "operation not supported on global/shared address space",
        ),
        718 => ("CUDA_ERROR_INVALID_PC", "invalid program counter"),
        719 => ("CUDA_ERROR_LAUNCH_FAILED", "unspecified launch failure"),
        720 => (
            "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE",
            "too many blocks in cooperative launch",
        ),
        721 => (
            "CUDA_ERROR_TENSOR_MEMORY_LEAK",
            "tensor memory not completely freed",
        ),
        800 => ("CUDA_ERROR_NOT_PERMITTED", "operation not permitted"),
        801 => ("CUDA_ERROR_NOT_SUPPORTED", "operation not supported"),
        802 => ("CUDA_ERROR_SYSTEM_NOT_READY", "system not yet initialized"),
        803 => (
            "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
            "system has unsupported display driver / cuda driver combination",
        ),
        804 => (
            "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE",
            "forward compatibility was attempted on non supported HW",
        ),
        805 => (
            "CUDA_ERROR_MPS_CONNECTION_FAILED",
            "MPS client failed to connect to the MPS control daemon or the MPS server",
        ),
        806 => (
            "CUDA_ERROR_MPS_RPC_FAILURE",
            "the remote procedural call between the MPS server and the MPS client failed",
        ),
        807 => (
            "CUDA_ERROR_MPS_SERVER_NOT_READY",
            "MPS server is not ready to accept new MPS client requests",
        ),
        808 => (
            "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED",
            "the hardware resources required to create MPS client have been exhausted",
        ),
        809 => (
            "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED",
            "the hardware resources required to support device connections have been exhausted",
        ),
        810 => (
            "CUDA_ERROR_MPS_CLIENT_TERMINATED",
            "the MPS client has been terminated by the server",
        ),
        811 => (
            "CUDA_ERROR_CDP_NOT_SUPPORTED",
            "is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it",
        ),
        812 => (
            "CUDA_ERROR_CDP_VERSION_MISMATCH",
            "unsupported interaction between different versions of CUDA Dynamic Parallelism",
        ),
        900 => (
            "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED",
            "operation not permitted when stream is capturing",
        ),
        901 => (
            "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED",
            "operation failed due to a previous error during capture",
        ),
        902 => (
            "CUDA_ERROR_STREAM_CAPTURE_MERGE",
            "operation would result in a merge of separate capture sequences",
        ),
        903 => (
            "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED",
            "capture was not ended in the same stream as it began",
        ),
        904 => (
            "CUDA_ERROR_STREAM_CAPTURE_UNJOINED",
            "capturing stream has unjoined work",
        ),
        905 => (
            "CUDA_ERROR_STREAM_CAPTURE_ISOLATION",
            "dependency created on uncaptured work in another stream",
        ),
        906 => (
            "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT",
            "operation would make the legacy stream depend on a capturing blocking stream",
        ),
        907 => (
            "CUDA_ERROR_CAPTURED_EVENT",
            "operation not permitted on an event last recorded in a capturing stream",
        ),
        908 => (
            "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD",
            "attempt to terminate a thread-local capture sequence from another thread",
        ),
        909 => ("CUDA_ERROR_TIMEOUT", "wait operation timed out"),
        910 => (
            "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE",
            "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update",
        ),
        911 => (
            "CUDA_ERROR_EXTERNAL_DEVICE",
            "an async error has occured in external entity outside of CUDA",
        ),
        912 => (
            "CUDA_ERROR_INVALID_CLUSTER_SIZE",
            "a kernel launch error has occurred due to cluster misconfiguration",
        ),
        913 => (
            "CUDA_ERROR_FUNCTION_NOT_LOADED",
            "the function handle is not loaded when calling an API that requires a loaded function",
        ),
        914 => (
            "CUDA_ERROR_INVALID_RESOURCE_TYPE",
            "one or more resources passed in are not valid resource types for the operation",
        ),
        915 => (
            "CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION",
            "one or more resources are insufficient or non-applicable for the operation",
        ),
        916 => (
            "CUDA_ERROR_KEY_ROTATION",
            "an error happened during the key rotation sequence",
        ),
        999 => ("CUDA_ERROR_UNKNOWN", "unknown error"),
        _ => ("unrecognized error code", ""),
    }
}
