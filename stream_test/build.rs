fn main() {
    cuda_min::GpuCode::new("gpu_ptx_code", "gpu_code").build()
}
