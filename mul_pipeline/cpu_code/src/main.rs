use std::time::Instant;

use cuda_min::{Device, Param};
fn main() {
    const A: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_code.ptx"));
    println!("PTX Code:");
    println!("{A}");
    let device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("vec_add").unwrap();

    const LEN: usize = 1024;
    let now = Instant::now();
    let mut ret = vec![0i32; LEN]; // ret could also be input, and its length decided the max avaliable tasks.
    let input1: Vec<_> = (0..LEN as i32).collect(); // normal parameter, its length is not restricted.
    let input2: Vec<_> = (0..LEN as i32).map(|x| x + 2).collect(); // normal parameter, its length is not restricted.
    let offset: Vec<_> = vec![1i32]; // You should restrict that, you never visit a range larger than the offset itself has

    let res = func
        .call(
            Param::new(&mut ret)
                .block_size(1024)
                .push(&input1)
                .push(&input2)
                .push(&offset)
                .shared(1024),
        )
        .unwrap();
    res.sync().unwrap();
    println!("{:?}", now.elapsed());
    ret.into_iter().enumerate().for_each(|(n, i)| {
        if i != input1[n] + input2[n] + offset[0] {
            println!("Add Error at {n}: {i}")
        }
    });
    println!("{:?}", now.elapsed());
}
