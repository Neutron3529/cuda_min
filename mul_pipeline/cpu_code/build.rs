fn main() {
    cuda_min::GpuCode::new("gpu_code", "../gpu_code")
        .target("../target")
        .build()
}
