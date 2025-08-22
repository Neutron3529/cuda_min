#![cfg(target_arch = "nvptx64")]
#![no_std]
#![no_main]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]

use core::arch::nvptx::*;
use nalgebra::Matrix3;

#[allow(unused_imports)] // for its panic-handler
use cuda_min;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn mat_mul(
    a: *const Matrix3<u8>,
    b: *const Matrix3<u8>,
    c: *mut Matrix3<u8>,
) {
    unsafe {
        let x = _thread_idx_x();
        let y = _thread_idx_y();
        let a = &*a;
        let b = &*b;
        let c = &mut *c;

        c[(x as usize, y as usize)] = (a.row(x as usize) * b.column(y as usize)).sum();
    }
}
