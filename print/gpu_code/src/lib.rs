#![cfg(target_arch = "nvptx64")]
#![no_std]
#![no_main]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vec_add(
    inout: *mut i32,
) {
    unsafe {
        vprintf("%d".as_ptr(), inout as *const _ as _);
    }
}
