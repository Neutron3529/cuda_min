use nalgebra::{Const, Matrix3};
use std::time::Instant;

use cuda_min::{Device, Param};
fn main() {
    const A: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gpu_ptx_code.ptx"));
    println!("PTX Code:");
    println!("{A}");
    let device = Device::init();

    let module = device.compile(A).unwrap();
    let func = module.get_function("mat_mul").unwrap();
    println!(
        "max thread per block is {}",
        func.get_max_thread_per_block().unwrap()
    );
    const LEN: usize = 1;
    let now = Instant::now();
    let mut ret = vec![Matrix3::<u8>::zeros(); LEN]; // ret could also be input, and its length decided the max avaliable tasks.
    let a = vec![
        Matrix3::<u8>::from_row_slice_generic(
            Const::<3>,
            Const::<3>,
            &[1u8, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        LEN
    ];
    let b = vec![
        Matrix3::<u8>::from_row_slice_generic(
            Const::<3>,
            Const::<3>,
            &[9u8, 8, 7, 6, 5, 4, 3, 2, 1]
        );
        LEN
    ];
    let mut param = Param::new(&mut ret).block_size(1).push(&a).push(&b);
    param = unsafe { param.set_block_grid_size((3, 3, 1), (1, 1, 1)) };

    let res = func.call(param).unwrap();
    res.sync().unwrap();
    println!("{:?}", now.elapsed());
    println!("{ret:?}");
    println!("{:?}", now.elapsed());
}
