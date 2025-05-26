#![cfg(target_arch = "nvptx64")]
#![no_std]
#![no_main]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn show_details(output: *mut (i32, i32, i32, i32)) {
    unsafe {
        let index = _block_idx_x() * _block_dim_x() + _thread_idx_x();
        *output.wrapping_add(index as usize) = (
            _grid_dim_x(),   // not change.
            _block_idx_x(),  // 0 .. _grid_dim_x()
            _block_dim_x(),  // not change
            _thread_idx_x(), // 0 .. _thread_idx_x()
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn show_details_align(output: *mut i32) {
    unsafe {
        let index = _block_idx_x() * _block_dim_x() + _thread_idx_x();
        let grid_dim = output.wrapping_add(index as usize);
        let gap = _grid_dim_x() * _block_dim_x();
        *grid_dim = _grid_dim_x();
        let block_idx = grid_dim.wrapping_add(gap as usize);
        *block_idx = _block_idx_x();
        let block_dim = block_idx.wrapping_add(gap as usize);
        *block_dim = _block_dim_x();
        let thread_idx = block_dim.wrapping_add(gap as usize);
        *thread_idx = _thread_idx_x();
    }
}
