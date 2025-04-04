use crate::Param;
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    num::NonZero,
};
// CUDA APIs
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CUerror(NonZero<c_int>);
pub type CUresult = Result<(), CUerror>;

impl core::fmt::Debug for CUerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA error: {}", self.0.get())
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUdevice(c_int);
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUcontext(*mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUmodule(*mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUfunction(*mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUstream(*mut c_void);

// 手动绑定 CUDA 驱动 API
#[link(name = "cuda")]
unsafe extern "C" {
    #[must_use = "You should check whether the execution successes."]
    fn cuInit(flags: c_uint) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    fn cuCtxCreate(ctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    fn cuModuleLoadData(module: *mut CUmodule, ptx: *const c_char) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    fn cuMemAlloc(dptr: *mut *mut c_void, bytesize: usize) -> CUresult;
    fn cuMemcpyHtoDAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    fn cuMemcpyDtoHAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult;
    // fn cuStreamCreate(stream: *mut CUstream, flag: c_uint) -> CUresult; // Parameters for stream creation (must be 0)
    #[must_use = "You should check whether the execution successes."]
    fn cuLaunchKernel(
        func: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem: c_uint,
        stream: CUstream,
        kernel_args: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    // fn cuCtxSynchronize() -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    fn cuStreamSynchronize(stream: CUstream) -> CUresult;
}
/// Cuda device and context
pub struct Device {
    #[allow(dead_code)]
    device: CUdevice,
    #[allow(dead_code)]
    context: CUcontext,
}
impl Device {
    const STREAM: CUstream = CUstream(std::ptr::null_mut()); // default null stream
    /// init device and context
    #[must_use = "You should check whether the execution successes."]
    pub fn init() -> Self {
        let mut device = CUdevice(0);
        let mut ctx = CUcontext(std::ptr::null_mut());
        unsafe {
            cuInit(0).unwrap();
            cuDeviceGet(&mut device, 0).unwrap();
            cuCtxCreate(&mut ctx, 0, device).unwrap();
        }
        Self {
            device,
            context: ctx,
        }
    }
    /// compile a module. Returns an error code 218 mostly means you do not send the correct PTX code into this function.
    #[must_use = "You should check whether the execution successes."]
    pub fn compile(&self, ptx: &str) -> Result<CUmodule, CUerror> {
        if let Ok(cstr) = &CString::new(ptx) {
            self.compile_raw(cstr)
        } else {
            Err(CUerror(NonZero::new(218).unwrap()))
        }
    }
    /// compile a module, with `&CStr` as its input
    #[must_use = "You should check whether the execution successes."]
    pub fn compile_raw(&self, c_ptx: &CStr) -> Result<CUmodule, CUerror> {
        let mut module = CUmodule(std::ptr::null_mut());
        unsafe { cuModuleLoadData(&mut module, c_ptx.as_ptr() as _)? }
        Ok(module)
    }
}
impl CUmodule {
    /// Get `CUfunction` from a module.
    #[must_use = "You should check whether the execution successes."]
    pub fn get_function(self, ptx: &str) -> Result<CUfunction, CUerror> {
        if let Ok(cstr) = &CString::new(ptx) {
            self.get_function_raw(cstr)
        } else {
            Err(CUerror(NonZero::new(218).unwrap()))
        }
    }
    /// Get `CUfunction` from a module, with `&CStr` as its input.
    #[must_use = "You should check whether the execution successes."]
    pub fn get_function_raw(self, function_name: &CStr) -> Result<CUfunction, CUerror> {
        let mut function = CUfunction(std::ptr::null_mut());
        unsafe { cuModuleGetFunction(&mut function, self, function_name.as_ptr())? }
        Ok(function)
    }
}

pub struct PendingResult<'a>([&'a (); 0]);
impl<'a> PendingResult<'a> {
    /// Wait for all the code finishes.
    #[must_use = "You should check whether the execution successes."]
    pub fn sync(self) -> CUresult {
        unsafe { cuStreamSynchronize(Device::STREAM) }
    }
}

impl CUfunction {
    /// Call a CUfunction, take care!
    /// SAFETY: You should check very careful since it is a ffi call, and it calls an unsafe function.
    /// You should notice that, this is not marked as unsafe, but you should always remember, this is not a safe function.
    #[must_use = "You should check whether the execution successes."]
    pub fn call<'a, R>(self, param: Param<'a, R>) -> Result<PendingResult<'a>, CUerror> {
        // SAFETY: Massive ffi calls.
        unsafe {
            let len = param.result.len();
            if len == 0 {
                // SAFETY in NonZero::new_unchecked: 1 != 0
                Err(CUerror(NonZero::new_unchecked(1)))?
            }
            let length = len * core::mem::size_of::<R>();
            let mut ret = core::ptr::null_mut();
            cuMemAlloc(&mut ret, length)?;
            cuMemcpyHtoDAsync(ret, param.result.as_ptr() as _, length, Device::STREAM)?;
            let mut device_mem = param
                .input
                .iter()
                .map(|_| core::ptr::null_mut())
                .collect::<Vec<_>>();
            for (i, size) in device_mem.iter_mut().zip(param.input.iter().map(|x| x.1)) {
                cuMemAlloc(i, size)?
            }
            for (device, host) in device_mem.iter().zip(param.input.iter()) {
                cuMemcpyHtoDAsync(*device, host.0, host.1, Device::STREAM)?
            }
            let mut device_ref = device_mem
                .iter_mut()
                .chain(core::iter::once(&mut ret))
                .map(|x| x as *mut _ as *mut c_void)
                .collect::<Vec<_>>();
            // println!("{self:?} {:?}", device_mem);
            cuLaunchKernel(
                self,
                param.grid_size.0,       // ----------
                param.grid_size.1,       // grid 维度
                param.grid_size.2,       // ----------
                param.block_size.0,      // ----------
                param.block_size.1,      // block 维度
                param.block_size.2,      // ----------
                param.shared_mem,        // 共享内存
                Device::STREAM,          // 流
                device_ref.as_mut_ptr(), // 参数指针
                core::ptr::null_mut(),
            )?;
            cuMemcpyDtoHAsync(param.result.as_mut_ptr() as _, ret, length, Device::STREAM)?
        }
        Ok(PendingResult([]))
    }
}
