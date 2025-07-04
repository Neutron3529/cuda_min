#![cfg(target_arch = "nvptx64")]
#![no_std]
#![no_main]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn random_access(input: *mut usize, output: *mut usize) {
    unsafe {
        let index = _block_idx_x() * _block_dim_x() + _thread_idx_x();
        let mut ai = index;
        for _ in 0..65536 {
            ai = *input.wrapping_add(ai as usize) as i32
        }
        *output.wrapping_add(index as usize) = ai as usize
    }
}
