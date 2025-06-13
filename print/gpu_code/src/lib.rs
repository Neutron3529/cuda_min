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
        vprintf("block %d thread %d print %d\n\0wtf?".as_ptr(), (&(_block_idx_x(), _thread_idx_x(), *inout.wrapping_add((_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize))) as *const _ as _);
    }
}
