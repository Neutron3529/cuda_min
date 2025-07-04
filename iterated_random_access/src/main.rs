#![feature(slice_as_array)]
use rayon::prelude::*;
use std::{hint::assert_unchecked as assume, time::Instant};

use cuda_min::{Device, Param};
fn init() -> Vec<usize> {
    let mut x = vec![0; 1048576];

    let mut curr = 0;
    for i in 0..1048576 {
        curr = (curr * 262145 + 24247759) % 1048576;
        x[i] = curr;
    }
    x
}
fn main() {
    // Init part
    let mut res_cpu = init();
    res_cpu.sort_unstable();
    res_cpu.iter().copied().enumerate().for_each(|(n, i)| {
        if n != i {
            println!("n={n}, i={i} mismatch")
        }
    });
    let a = init();
    // Parallel CPU part
    let now = Instant::now();
    (0..1048576)
        .into_par_iter()
        .zip(&mut res_cpu)
        .chunks(1024)
        .for_each(|x| {
            for i in x {
                let mut ai = i.0;
                unsafe { assume(i.0 < a.len()) }
                for _fghfffff in 0..65536 {
                    unsafe { assume(ai < a.len()) }
                    ai = a[ai]
                }
                *i.1 = ai
            }
        });
    println!("got time = {:?}", now.elapsed());
    // // CPU part
    // let now = Instant::now();
    // for i in 0..1048576 {
    //     unsafe { assume(i < a.len()) }
    //     let mut ai = i;
    //     for _ in 0..4096 {
    //         unsafe { assume(ai < a.len()) }
    //         ai = a[ai]
    //     }
    //     res_cpu[i] = ai
    // }
    // println!("calc time = {:?}", now.elapsed());
    // GPU part
    const A: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_ptx_code.ptx"));
    println!("PTX Code:");
    println!("{A}");
    let device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("random_access").unwrap();
    println!(
        "max thread per block is {}",
        func.get_max_thread_per_block().unwrap()
    );
    const LEN: usize = 1048576;
    let now = Instant::now();
    let mut ret = vec![0usize; LEN]; // ret could also be input, and its length decided the max avaliable tasks.
    let param = Param::new(&mut ret).block_size(1024).push(a.as_slice());

    let res = func.call(param).unwrap();
    res.sync().unwrap();
    println!("{:?}", now.elapsed());
    ret.into_iter()
        .zip(res_cpu.into_iter())
        .enumerate()
        .for_each(|(n, i)| {
            if i.0 != i.1 {
                println!("Add Error at {n}: {} should be {}", i.0, i.1)
            }
        });
    println!("{:?}", now.elapsed());
}
