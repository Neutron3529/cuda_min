use cuda_min::{Device, Param};
const PTX:&'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_ptx.ptx"));
fn main() {
    let now = std::time::Instant::now();
    let device = Device::init();
    // debug(device);return;
    let module = device.compile(PTX).unwrap();
    let func = module.get_function("mont_pows_batch").unwrap();
    println!("Init cost {:?}", now.elapsed());
    let now = std::time::Instant::now();
    let mut ret = (0u64..10310).collect::<Vec<_>>();
    let param = Param::new(&mut ret).block_size(1024);
    println!(
        "max thread per block is {}",
        func.get_max_thread_per_block().unwrap()
    );
    // param.grid_size = 1;
    // param.block_size = 12;
    // param.push(&input);
    // println!("{param:?}");
    let res = func.call(param).unwrap();
    res.sync().unwrap();
    ret.into_iter().enumerate().for_each(|(n, i)| {
        if i != 0 {
            println!("block {n} got {i} non-zero results")
        }
    });
    println!("{:?}", now.elapsed());
}
#[allow(unused)]
fn debug(device: Device) {
    let module = device.compile(PTX).unwrap();
    let func = module.get_function("mont_pows_debug_181251_37").unwrap();
    let mut ret = [[[0u64; 2]; 64]];
    let param = Param::new(&mut ret);
    let res = func.call(param).unwrap();
    res.sync().unwrap();
    println!("{ret:?}")
}
