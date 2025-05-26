#![no_main]
#![cfg(target_arch = "nvptx64")]
#![no_std]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vec_add(
    input1: *const i32,
    input2: *const i32,
    offset: *const i32,
    output: *mut i32,
) {
    unsafe {
        let offset = *offset;
        let index = _block_idx_x() * _block_dim_x() + _thread_idx_x();
        let left = *input1.wrapping_add(index as usize);
        let right = *input2.wrapping_add(index as usize);
        *output.wrapping_add(index as usize) = left + right + offset;

        vprintf("wtf?\0".as_ptr(), core::ptr::null());
    }
}
