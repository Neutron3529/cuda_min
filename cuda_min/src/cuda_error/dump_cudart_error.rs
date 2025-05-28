use core::{
    mem,
    ffi::{CStr, c_int},
    ptr,
};

#[cfg(feature = "native-error-desc")]
use crate::CUresult;
#[cfg(not(feature = "native-error-desc"))]
type CUresult = c_int;
#[cfg(not(feature = "native-error-desc"))]
trait CUresultIsOk {
    fn is_ok(self) -> bool;
}
#[cfg(not(feature = "native-error-desc"))]
impl CUresultIsOk for CUresult {
    fn is_ok(self) -> bool {
        self == 0
    }
}

const _ASSERT_SIZE_EQUAL: () = assert!(
    mem::size_of::<CUresult>() == mem::size_of::<c_int>(),
    "CUresult must be c_int"
);

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuGetErrorName(error_code: CUresult, name_ptr: &mut *const i8) -> CUresult;
    pub fn cuGetErrorString(error_code: CUresult, desc_ptr: &mut *const i8) -> CUresult;
}
const UNKNOWN_ERROR_NAME: &'static CStr = c"unrecognized error code";
const UNKNOWN_ERROR_DESC: &'static CStr = c"";

pub fn cu_get_error_name(error: CUresult) -> &'static CStr {
    unsafe {
        let ptr = &mut ptr::null();
        if cuGetErrorName(error, ptr).is_ok() {
            CStr::from_ptr(*ptr)
        } else {
            UNKNOWN_ERROR_NAME
        }
    }
}

pub fn cu_get_error_string(error: CUresult) -> &'static CStr {
    unsafe {
        let ptr = &mut ptr::null();
        if cuGetErrorString(error, ptr).is_ok() {
            CStr::from_ptr(*ptr)
        } else {
            UNKNOWN_ERROR_DESC
        }
    }
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
#[cfg(not(feature = "native-error-desc"))]
fn main() {
    use std::{
        fs::File,
        io::{BufWriter, Write},
    };
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
    let unrecognized = cu_get_error_name(unrecognized);
    let mut file = BufWriter::new(File::create(args.get(2).map_or("name_desc.rs", |l| l)).unwrap());
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
        let name = cu_get_error_name(i);
        if name != unrecognized {
            writeln!(
                file,
                r#"        {i} => ({name:?}, {:?}),"#,
                cu_get_error_string(i)
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
