#![no_std]
#![no_main]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vec_add(input1: *mut i32, input2: *mut i32, output: *mut i32) {
    unsafe {
        let index = _block_idx_x() * _block_dim_x() + _thread_idx_x();
        *output.wrapping_add(index as usize) =
            *input1.wrapping_add(index as usize) + *input2.wrapping_add(index as usize)
    }
}
