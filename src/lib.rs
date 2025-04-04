//! # cuda_min: A small crate that could run CUDA program (write in PTX format) easily.
//!
//! This is a very simple crate that just contains essential functions and provides a way
//! to run your rust code on nvidia GPU.
//!
//! Currently this crate just provides a simple way to interact with PTX code, and this
//! crate might be better regarded as a tutorial for running rust code in your nvidia GPU.
//!
//! # Usage
//!
//! Actually this crate provides NO methods for nvptx64 backend. This crate is mainly for
//! running specific PTX code (that receives parameters and returns only its last parameter).
//! Since it might not be easy to provide such code, I'll write a brief guide:
//!
//! ## 1. Minimal example:
//!
//! You must have some valid ptx code, for example, manually write it down:
//! ```
//! .visible .entry number_off(
//! .param .u64 .ptr .align 1 counter_param_0
//! )
//! {
//!     .reg .b32 	%r<5>;
//!     .reg .b64 	%rd<5>;
//!
//!     // %bb.0:
//!     ld.param.u64 	%rd1, [counter_param_0];
//!     cvta.to.global.u64 	%rd2, %rd1;
//!     mov.u32 	%r1, %ctaid.x;
//!     mov.u32 	%r2, %ntid.x;
//!     mov.u32 	%r3, %tid.x;
//!     mad.lo.s32 	%r4, %r1, %r2, %r3;
//!     mul.wide.s32 	%rd3, %r4, 4;
//!     add.s64 	%rd4, %rd2, %rd3;
//!     st.global.u32 	[%rd4], %r4;
//!     ret;
//!     // -- End function
//! }
//! ```
//! With such code is stored as "ptx.s", you could use this crate to call the function and read its returned values.
//! ```
//! use cuda::*;
//! fn main() {
//!     let func = cuda::compile(include_str!("ptx.s")).unwrap().get_function("number_off").unwrap();
//!     let mut ret = [0u32;128];
//!     let mut param = Param::new(&mut ret); // Here the block size and grid size is calculated automatically.
//!     func.call(param).unwrap().sync().unwrap();
//!     println!("{:?}", ret); // You should get a vector with 0,1,...,127.
//! }
//! ```
//! You might notice that, there is no unsafe here, but actually all functions are unsafe. Since you're calling cuda function, there is **actually no safety** here. No need to mark the whole function as unsafe.
//!
//! ## Generates ptx code (for example, with nvptx64 backend)
//!
//! Although you might write ptx code directly, and you might mainly writting ptx asm in rust code
//! due to performance issues. It might still be better to write rust code to generate PTX code.
//!
//! Since rust could be regarded as a good-enough asm auto expander, writting rust code is still fruitful.
//! And since `core::arch::nvptx64` is an experimental backend, writting and compiling nvptx64 code could not be done automatically (unless build.rs/xtask/... is used)
//!
//! To generate PTX code, you should firstly set the default target and default compile args for nvptx64:
//! ```
//! # in ~/.cargo/config, since it is default for all nvptx64 target
//! [target.'cfg(target_arch="nvptx64")']
//! rustflags = [
//!     "-C", "target-cpu=sm_86", # sm_86 is what suits for my computer (nvidia 3060 on my Laptop, you should change it with what `cudaGetDeviceProperties` provides)
//!     # uncomment only one of the following lines:
//!     # "-C", "linker=llvm-bitcode-linker" # yields `crate-name.ptx`, but very slow. Needs to install llvm-bitcode-linker
//!     # "--emit=asm"                       # fast, but yields `crate-name-{hash}.s`. Needs to execute cargo clean otherwise the .s might not be yielded.
//! ]
//! # in your crate that targets `nvptx64`
//! [build]
//! target = "nvptx64-nvidia-cuda"
//! [rust-analyzer]
//! cargo.target = "nvptx64-nvidia-cuda"
//! ```
//!
//! In case you don't know what `sm_` your gpu is, but you have installed cuda and have a cpp compiler, you could use this simple program:
//! ```
//! #include <stdio.h>
//! #include <cuda_runtime.h>
//!
//! int main() {
//!     int device_count;
//!     cudaGetDeviceCount(&device_count);
//!     for (int i = 0; i < device_count; i++) {
//!         cudaDeviceProp prop;
//!         cudaGetDeviceProperties(&prop, i);
//!         printf("GPU %d: sm_%d%d\n", i, prop.major, prop.minor);
//!     }
//!     return 0;
//! }
//! ```
//!
//! In case you have configured config.toml, you could directly use `cargo build --release` to compile rust code into PTX code.
//!
//! One of the exception might be, a `#[panic_handler]` is needed, in this case, you could write a simple one to makes rust happy:
//! ```
//! #[panic_handler] fn panic(_:&core::panic::PanicInfo) ->! { loop{} }
//! ```
//!

mod cuda;
pub use cuda::Device;
use cuda::*;
pub fn compile(s: &str) -> Result<CUmodule, CUerror> {
    Device::init().compile(s)
}
/// Function Param
///
/// Packing and transfering parameters and returns to and from GPU.
#[derive(Debug)]
pub struct Param<'a, R> {
    /// input pointers and lengths (in bytes)
    pub input: Vec<(*const core::ffi::c_void, usize)>,
    /// result, controls the numbers of tasks
    pub result: &'a mut [R],
    /// shared memory size, I have not yet tested it.
    pub shared_mem: u32,
    block_size: (u32, u32, u32),
    grid_size: (u32, u32, u32),
}
impl<'a, R> Param<'a, R> {
    /// Set block 1d size. Currently only 1d size could be set directly (since result is a 1d vector)
    pub fn block_size(&mut self, val: u32) {
        self.block_size = (val, 1, 1);
        self.grid_size = (
            (self.result.len() / self.block_size.0 as usize) as u32,
            1,
            1,
        );
    }
    /// Set block size and grid size. You must ensure you handled it very well.
    /// Although all the functions are not very safe. This function is extremely unsafe.
    pub unsafe fn set_block_grid_size(
        &mut self,
        block_size: (u32, u32, u32),
        grid_size: (u32, u32, u32),
    ) {
        self.block_size = block_size;
        self.grid_size = grid_size;
    }
    /// generate parameter from its output, use output's length as number of tasks.
    pub fn new(result: &'a mut [R]) -> Self {
        let block_size = (
            32.max(((result.len() >> 32).max(1)).ilog(2) + 1) as u32,
            1,
            1,
        );
        let grid_size = ((result.len() / block_size.0 as usize) as u32, 1, 1);
        Self {
            input: Vec::new(),
            result,
            shared_mem: 0,
            block_size,
            grid_size,
        }
    }
    /// push vectors into this parameter collection.
    pub fn push<T>(&mut self, item: &[T]) -> bool {
        // let size = core::mem::size_of_val(item);
        if let Some(size) = core::num::NonZero::new(core::mem::size_of::<T>() * item.len()) {
            self.input.push((item.as_ptr() as _, size.get()));
            true
        } else {
            false
        }
    }
}
