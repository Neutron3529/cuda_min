#![cfg(target_arch = "nvptx64")]
#![no_std]
#![no_main]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn data_processor(input: *const i64, output: *mut i64) {
    unsafe {
        let index = _block_idx_x() * _block_dim_x() + _thread_idx_x();
        *output.wrapping_add(index as usize) = ((index as i64) << 32) + (index as i64 % *input);
    }
}
