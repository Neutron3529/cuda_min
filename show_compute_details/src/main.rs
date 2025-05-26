use std::time::Instant;

use cuda_min::{Device, Param};
fn main() {
    const A: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_ptx_code.ptx"));
    println!("PTX Code:");
    println!("{A}");
    let device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("show_details").unwrap();
    println!(
        "max thread per block is {}",
        func.get_max_thread_per_block().unwrap()
    );
    let func_align = module.get_function("show_details_align").unwrap();
    println!(
        "max thread per block is {}",
        func_align.get_max_thread_per_block().unwrap()
    );
    const LEN: usize = 4096;
    let now = Instant::now();
    let mut ret = vec![(0i32, 0i32, 0i32, 0i32); LEN]; // ret could also be input, and its length decided the max avaliable tasks.
    let mut param = Param::new(&mut ret).block_size(1024).len(LEN);

    let res = func.call(param).unwrap();
    res.sync().unwrap();
    println!("{:?}", now.elapsed());
    ret.iter()
        .enumerate()
        .for_each(|(n, i)| println!("index {n}: {i:?}"));
    println!("{:?}", now.elapsed());
    println!("loop ..");

    let mut ret2 = vec![[0i32; LEN]; 4];
    let mut param2 = unsafe { Param::new(&mut ret2).block_size(1024).len(LEN) };
    let now = Instant::now();
    println!("loop ...");
    let res2 = func_align.call(param2).unwrap().sync().unwrap();
    println!("loop ....");
    println!("align cost {:?}", now.elapsed());
    const SIZE: usize = 1048576;
    let mut ret = vec![(0i32, 0i32, 0i32, 0i32); SIZE]; // ret could also be input, and its length decided the max avaliable tasks.
    let mut ret2 = vec![0i32; SIZE * 4]; // ret could also be input, and its length decided the max avaliable tasks.
    for i in 0..10 {
        println!("loop {i}");
        let now = Instant::now();
        let mut param = unsafe { Param::new(&mut ret).len(SIZE).block_size(1024) };
        let res2 = func.call(param).unwrap().sync().unwrap();
        println!("func cost {:?}", now.elapsed());
        let now = Instant::now();
        let mut param2 = unsafe { Param::new(&mut ret2).len(SIZE).block_size(1024) };
        let res2 = func_align.call(param2).unwrap().sync().unwrap();
        println!("align cost {:?}", now.elapsed());
    }
    println!("aligned code seems to be faster.")
}
