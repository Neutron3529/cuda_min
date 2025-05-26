#[cfg(feature = "cudart")]
use crate::CUresult;
#[cfg(not(feature = "cudart"))]
type CUresult = std::ffi::c_int;

#[link(name = "cudart")]
unsafe extern "C" {
    pub fn cudaGetErrorName(error: CUresult) -> *const i8;
    pub fn cudaGetErrorString(error: CUresult) -> *const i8;
}

/**
 * Usage:
 *
```bash
LIBRARY_PATH=$CUDA_PATH/lib rustc --edition 2024 dump_cudart_error.rs -o dump_cudart_error && ./dump_cudart_error && rustfmt --style-edition 2024 name_desc.rs && rm ./dump_cudart_error
```
 *
 * Notice that, this file behind #[cfg(feature = "cudart")], thus logically with #[cfg(not(feature = "cudart"))], fn main will never being compiled.
 *
 * In case you want this main file, you should regard this file as a single .rs file, using bash script above to compile and execute.
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
