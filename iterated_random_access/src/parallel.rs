use std::{hint::assert_unchecked as assume, time::Instant};

use cuda_min::{Device, Param};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
fn init() -> Box<[usize; 1048576]> {
    let mut x = Box::new([0; 1048576]);
    let mut curr = 0;
    for i in 0..1048576 {
        curr = (curr * 262145 + 24247759) % 1048576;
        x[i] = curr
    }
    x
}
fn main() {
    let mut a = init();
    a.sort_unstable();
    a.iter().copied().enumerate().for_each(|(n, i)| {
        if n != i {
            println!("n={n}, i={i} mismatch")
        }
    });
    let a = init();
    let now = Instant::now();
    let res = (0..1048576)
        .into_par_iter()
        .chunks(1024)
        .map(|x| {
            let mut res = 0;
            for i in x {
                unsafe { assume(i < a.len()) }
                let mut ai = i;
                for i in 0..65536 {
                    unsafe { assume(ai < a.len()) }
                    ai = a[i]
                }
                res ^= ai
            }
            res
        })
        .reduce(|| 0, |x, y| x ^ y);
    println!("got result = {res}, time = {:?}", now.elapsed())
    /*
    const A: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_ptx_code.ptx"));
    println!("PTX Code:");
    println!("{A}");
    let device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("vec_add").unwrap();
    println!(
        "max thread per block is {}",
        func.get_max_thread_per_block().unwrap()
    );
    const LEN: usize = 1048576;
    let now = Instant::now();
    let mut ret = vec![0i32; LEN]; // ret could also be input, and its length decided the max avaliable tasks.
    let input1: Vec<_> = (0..LEN as i32).collect(); // normal parameter, its length is not restricted.
    let input2: Vec<_> = (0..LEN as i32).map(|x| x + 2).collect(); // normal parameter, its length is not restricted.
    let offset: Vec<_> = vec![1i32]; // You should restrict that, you never visit a range larger than the offset itself has
    let mut param = Param::new(&mut ret)
        .block_size(1024)
        .push(&input1)
        .push(&input2)
        .push(&offset);

    let res = func.call(param).unwrap();
    res.sync().unwrap();
    println!("{:?}", now.elapsed());
    ret.into_iter().enumerate().for_each(|(n, i)| {
        if i != input1[n] + input2[n] + offset[0] {
            println!("Add Error at {n}: {i}")
        }
    });
    println!("{:?}", now.elapsed());
    */
}
