use std::time::Instant;

use cuda_min::{Device, Param};
fn main() {
    const A: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_ptx_code.ptx"));
    println!("PTX Code:");
    println!("{A}");
    let device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("data_processor").unwrap();
    println!(
        "max thread per block is {}",
        func.get_max_thread_per_block().unwrap()
    );
    const LEN: usize = 1048576;
    for _ in 0..10 {
        let now = Instant::now();
        let mut ret = vec![0i128; LEN]; // ret could also be input, and its length decided the max avaliable tasks.
        let input: Vec<_> = (1..=LEN as i64).collect(); // normal parameter, its length is not restricted.
        let param = Param::new(&mut ret).block_size(1024).push(&input);

        let res = func.call(param).unwrap();
        res.sync().unwrap();
        println!("{:?}", now.elapsed());
        ret.into_iter().enumerate().for_each(|(n, i)| {});
        println!("{:?}", now.elapsed());
    }
}
