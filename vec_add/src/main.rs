use std::time::Instant;

use cuda_min::{Device, Param};
fn main() {
    const A: &'static str =
        include_str!("../gpu_code/target/nvptx64-nvidia-cuda/release/gpu_code.ptx");
    println!("PTX Code:");
    println!("{A}");
    let mut device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("vec_add").unwrap();

    const LEN: usize = 1024;
    let now = Instant::now();
    let mut ret = vec![0i32; LEN]; // ret could also be input, and its length decided the max avaliable tasks.
    let input1: Vec<_> = (0..LEN as i32).collect(); // ret could also be input, and its length decided the max avaliable tasks.
    let input2: Vec<_> = (0..LEN as i32).map(|x| x + 2).collect(); // ret could also be input, and its length decided the max avaliable tasks.
    let mut param = Param::new(&mut ret);
    param.block_size(1024);
    param.push(&input1);
    param.push(&input2);

    let res = func.call(param).unwrap();
    res.sync().unwrap();
    ret.into_iter().enumerate().for_each(|(n, i)| {
        if i != 2 * n as i32 + 2 {
            println!("Add Error at {n}: {i}")
        }
    });
    println!("{:?}", now.elapsed());
}
