#[cfg(feature = "cudart")]
use crate::CUresult;
use std::{
    ffi::CStr,
    fs::File,
    io::{BufWriter, Write},
};
#[cfg(not(feature = "cudart"))]
type CUresult = std::ffi::c_int;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaGetErrorName(error: CUresult) -> *const i8;
    fn cudaGetErrorString(error: CUresult) -> *const i8;
}

/**
```bash
export LIBRARY_PATH=$CUDA_PATH/bin && export LD_LIBRARY_PATH=$CUDA_PATH/bin && rustc --edition 2024 dump_cudart_error.rs -o dump_cudart_error && ./dump_cudart_error && rm ./dump_cudart_error
```
*/

#[cfg(not(feature = "cudart"))]
fn main() {
    unsafe {
        let args: Vec<_> = std::env::args().collect();
        let to = args
            .get(1)
            .map(|x| x.parse().ok())
            .flatten()
            .unwrap_or(10000);
        let unrecognized = args
            .get(2)
            .map(|x| x.parse().ok())
            .flatten()
            .unwrap_or(9999);
        let unrecognized = CStr::from_ptr(cudaGetErrorName(unrecognized));
        let mut file =
            BufWriter::new(File::create(args.get(2).map_or("name_desc.rs", |l| l)).unwrap());
        // let name = args.get(2).map_or("name_desc.map",|l|l);
        // let append = false; // args.get(3).is_some();
        // let opt = &mut OpenOptions::new();
        // let opt = opt.create(true).write(true).append(append);
        // let [mut namef, mut descf] = [name, desc].map(|x| BufWriter::new(opt.open(x).unwrap()));
        writeln!(
            file,
            "pub fn get_name_desc(code: std::ffi::c_int) -> (&'static str, &'static str) {{
    match code {{"
        )
        .unwrap();
        for i in 0..=to {
            let name = CStr::from_ptr(cudaGetErrorName(i));
            if name != unrecognized {
                writeln!(
                    file,
                    r#"        {i} => ({name:?}, {:?}),"#,
                    CStr::from_ptr(cudaGetErrorString(i))
                )
                .unwrap();
            }
        }
        writeln!(
            file,
            r#"        _ => ({unrecognized:?}, "")
    }}
}}"#
        )
        .unwrap()
    }
}
