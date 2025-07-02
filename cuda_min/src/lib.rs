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
//! You must have some valid ptx code. In case you have it, you could use this crate to call the function and read its returned values.
//! ```
//! use cuda_min::{Device, Param};
//! fn main() {
//!     let ptx = r#"
//! .version 7.1
//! .target sm_30
//! .address_size 64
//! .visible .entry number_off(
//!     .param .u64 .ptr .align 1 counter_param_0
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
//! }"#; // A better solution is save this file in `ptx.s` and then using `include_str!("ptx.s")` to load it directly.
//!     println!("{ptx}");
//!     let device = Device::init(); // fast init, use the first GPU only. Panic if no GPU is provided.
//!     let module = device.compile(ptx).unwrap();
//!     let func = module.get_function("number_off").unwrap();
//!     let mut ret = [0u32;128];
//!     let mut param = cuda_min::Param::new(&mut ret); // Here the block size and grid size is calculated automatically.
//!     func.call(param)
//!         .unwrap()
//!         .sync()
//!         .unwrap();
//!     assert_eq!((0..128).collect::<Vec<_>>(), ret); // You should get a vector with 0,1,...,127.
//! }
//! ```
//! You might notice that, there is no unsafe here, but actually all functions are unsafe. Since you're calling cuda function, there is **actually no safety** here. No need to mark the whole function as unsafe.
//!
//! ## Generates ptx code (for example, with nvptx64 backend)
//!
//! Although you might write ptx code directly, and you might mainly writting ptx asm in Rust code
//! due to performance issues. It might still be better to write rust code to generate PTX code.
//!
//! Since rust could be regarded as a good-enough asm auto expander, writting rust code is still fruitful.
//! And since `core::arch::nvptx64` is an experimental backend, writting and compiling nvptx64 code could not be done automatically (unless build.rs/xtask/... is used)
//!
//! To generate PTX code, you should firstly set the default target and default compile args for nvptx64:
//! ```toml
//! # In `~/.cargo/config.toml`, since it is default for all nvptx64 target
//! [target.'cfg(target_arch="nvptx64")']
//! rustflags = [
//!     "-C", "target-cpu=sm_86", # sm_86 is what suits for my computer (nvidia 3060 on my Laptop, you should change it with what `cudaGetDeviceProperties` provides)
//!     ### uncomment only one of the following lines:
//!     # "-C", "linker=llvm-bitcode-linker" # yields `crate-name.ptx`, but very slow. Needs to install `llvm-bitcode-linker` and its dependencies, e.g., `llvm-tools`
//!     # "--emit=asm"                       # fast, but yields `deps/crate-name-{hash}.s`. Needs to execute cargo clean otherwise the .s might not be yielded.
//!     ### both of the above code generates almost the same code.
//! ]
//! ```
//! ```toml
//! # In `.cargo/config.toml`, your crate that targets `nvptx64`
//! [build]
//! target = "nvptx64-nvidia-cuda"
//! [rust-analyzer]
//! cargo.target = "nvptx64-nvidia-cuda"
//! ```
//! ```toml
//! #  In`Cargo.toml`, your crate that targets `nvptx64`, if `llvm-bitcode-linker` is chosen:
//! [lib]
//! crate-type = ["cdylib"] # comment this line if you choose `--emit-asm` mode, since it might try call `rust-ptx-linker`
//! ```
//!
//! In case you don't know what `sm_` your gpu is, you could use this simple program:
//! ```
//! use cuda_min::Device;
//! fn main() {
//!     println!("{}", Device::init().get_native_target_cpu().unwrap()) // there is a cuda-tool called `nvptx_arch`, you could also use this to obtain the correct GPU architecture.
//! }
//! ```
//!
//! In case you have configured config.toml, you could directly use `cargo build --release` to compile rust code into PTX code.
//!
//! One of the exception might be, a `#[panic_handler]` is needed, in this case, you could write a simple one to makes rust happy:
//! ```
//! // In fact, this crate already provides such function for target_arch = "nvptx64" with feature "panic_handler". In case you have your own panic handler, just disable that feature.
//! #[cfg_attr(target_arch = "nvptx64", panic_handler)] fn panic(_:&core::panic::PanicInfo) ->! { loop{} }
//! ```
#![cfg_attr(target_arch = "nvptx64", no_std, feature(asm_experimental_arch, stdarch_nvptx))]
#![feature(trace_macros)]

#[macro_export]
macro_rules! repeat {
    (@ $b:block $t:tt) => { $b };
    ({ $($t:tt)* } $b:block) => { $(repeat!{@ $b $t})* };
}

#[cfg(target_arch = "nvptx64")]
#[cfg_attr(all(target_arch = "nvptx64", feature = "panic-handler"), panic_handler)]
unsafe fn ph(_info: &core::panic::PanicInfo) -> ! {
    use core::arch::nvptx::{_thread_idx_x, _thread_idx_y, _thread_idx_z, _block_idx_x, _block_idx_y, _block_idx_z};
    unsafe {
        #[cfg(feature = "using_v2_suffix")]
        if let Some(message) = _info.message().as_str() {
            vprintf(c"block (%d, %d, %d) thread (%d, %d, %d) panics: %s\n", message, _block_idx_x(), _block_idx_y(), _block_idx_z(), _thread_idx_x(), _thread_idx_y(), _thread_idx_z());
        } else {
            core::arch::nvptx::vprintf(c"block (%d, %d, %d) thread (%d, %d, %d): panic occors.\n".as_ptr() as _, &(_block_idx_x(), _block_idx_y(), _block_idx_z(), _thread_idx_x(), _thread_idx_y(), _thread_idx_z()) as *const _ as _);
        }
        #[cfg(feature = "using_v2_suffix")]
        let caller = _info.location();
        if let Some(loc) = caller {
            vprintf_file_loc(c"block (%d, %d, %d) thread (%d, %d, %d):  at file: %s, line: %d column: %d\n", loc.file(), loc.line(), loc.column(), _block_idx_x(), _block_idx_y(), _block_idx_z(), _thread_idx_x(), _thread_idx_y(), _thread_idx_z());
        }
        abort()
    }
}

/// wrapper of vprintf.
/// format should have exact one "%s" after 6 leadin "%d"s.
/// message ends with '\0' is better than without '\0'. There should be exact 1 '\0' in message.
#[inline(always)]
#[cfg(target_arch = "nvptx64")]
pub unsafe fn vprintf(format: &core::ffi::CStr, message: &str, bx: i32, by: i32, bz: i32, tx: i32, ty: i32, tz: i32) {
    #[repr(C)]
    struct Message(i32, i32, i32, i32, i32, i32,*const u8);
    if message.ends_with('\0') {
        unsafe { core::arch::nvptx::vprintf(format.as_ptr() as _, &Message(bx,by,bz,tx,ty,tz,message.as_ptr()) as *const _ as  _); }
    } else {
        const BUFFER_MAX: usize = 1024;
        const TRUNCATE: [u8;4] = *b"...\0";
        let mut buffer = [0u8;BUFFER_MAX];
        unsafe {
            if message.len() < BUFFER_MAX {
                core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr(), message.len());
                buffer[message.len()] = 0
            }
            else {
                core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr(), BUFFER_MAX - TRUNCATE.len());
                core::ptr::copy_nonoverlapping(b"...\0".as_ptr(), buffer.as_mut_ptr().wrapping_add(BUFFER_MAX - TRUNCATE.len()), TRUNCATE.len());
            }
            core::arch::nvptx::vprintf(format.as_ptr() as _, &Message(bx,by,bz,tx,ty,tz,buffer.as_ptr()) as *const _ as  _);
        }
    }
}

/// wrapper of vprintf.
/// format should have the format: .. %d .. %d .. %d .. %d .. %d .. %d ..%s .. %d .. %d
#[inline(always)]
#[cfg(target_arch = "nvptx64")]
pub unsafe fn vprintf_file_loc(format: &core::ffi::CStr, message: &str, line: u32, column: u32, bx: i32, by: i32, bz: i32, tx: i32, ty: i32, tz: i32) {
    #[repr(C)]
    struct Message(i32, i32, i32, i32, i32, i32, *const u8, u32, u32);
    const BUFFER_MAX: usize = 1024;
    const TRUNCATE: [u8;4] = *b"...\0";
    let mut buffer = [0u8;BUFFER_MAX];
    unsafe {
        if message.len() < BUFFER_MAX {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr(), message.len());
            buffer[message.len()] = 0
        }
        else {
            core::ptr::copy_nonoverlapping(message.as_ptr(), buffer.as_mut_ptr(), BUFFER_MAX - TRUNCATE.len());
            core::ptr::copy_nonoverlapping(b"...\0".as_ptr(), buffer.as_mut_ptr().wrapping_add(BUFFER_MAX - TRUNCATE.len()), TRUNCATE.len());
        }
        core::arch::nvptx::vprintf(format.as_ptr() as _, &Message(bx,by,bz,tx,ty,tz,buffer.as_ptr(), line, column) as *const _ as  _);
    }
}


#[inline(always)]
#[cfg(target_arch = "nvptx64")]
pub fn abort() -> ! {
    unsafe { core::arch::asm!("trap;", options(noreturn)) }
}

#[cfg(not(target_arch = "nvptx64"))]
mod host;
#[cfg(not(target_arch = "nvptx64"))]
pub use host::*;
#[cfg(all(
    feature = "build-script-with-llvm-bitcode-linker",
    not(target_arch = "nvptx64")
))]
mod build;
#[cfg(all(
    feature = "build-script-with-llvm-bitcode-linker",
    not(target_arch = "nvptx64")
))]
pub use build::GpuCode;
