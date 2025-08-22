#![cfg(target_arch = "nvptx64")]
#![no_std]
#![no_main]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vec_add(
    input1: *const i128,
    input2: *const i128,
    offset: *const i128,
    output: *mut i128,
) {
    unsafe {
        let offset = *offset;
        let index = _block_idx_x() * _block_dim_x() + _thread_idx_x();
        let left = *input1.wrapping_add(index as usize);
        let right = *input2.wrapping_add(index as usize);
        *output.wrapping_add(index as usize) = left * right + offset;
    }
}
