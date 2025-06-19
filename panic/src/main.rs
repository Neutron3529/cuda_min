use std::time::Instant;

use cuda_min::{Device, Param};
fn main() {
    const A: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_ptx_code.ptx"));
    println!("PTX Code:");
    println!("{A}");
    let device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("may_panic").unwrap();
    println!(
        "max thread per block is {}",
        func.get_max_thread_per_block().unwrap()
    );
    const LEN: usize = 1024;
    let now = Instant::now();
    let mut ret = (0..LEN as i32).collect::<Vec<_>>(); // ret could also be input, and its length decided the max avaliable tasks.
    let mut param = Param::new(&mut ret)
        .block_size(1024);

    let res = func.call(param).unwrap();
    res.sync().unwrap();
    println!("{:?}", now.elapsed());
}
