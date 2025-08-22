use std::time::Instant;

use cuda_min::{Device, Param};
fn main() {
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
    let mut ret = vec![0i128; LEN]; // ret could also be input, and its length decided the max avaliable tasks.
    let input1: Vec<_> = (0..LEN as i128).collect(); // normal parameter, its length is not restricted.
    let input2: Vec<_> = (0..LEN as i128).map(|x| x + 2).collect(); // normal parameter, its length is not restricted.
    let offset: Vec<_> = vec![1i128 << 64]; // You should restrict that, you never visit a range larger than the offset itself has
    let param = Param::new(&mut ret)
        .block_size(1024)
        .push(&input1)
        .push(&input2)
        .push(&offset);

    let res = func.call(param).unwrap();
    res.sync().unwrap();
    println!("{:?}", now.elapsed());
    ret.into_iter().enumerate().for_each(|(n, i)| {
        if i != input1[n] * input2[n] + offset[0] {
            println!("Add Error at {n}: {i}")
        }
    });
    println!("{:?}", now.elapsed());
}
