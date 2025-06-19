use crate::Param;
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    fmt, iter,
    marker::PhantomData,
    mem,
    num::NonZero,
    ptr,
};

// CUDA APIs
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CUerror(NonZero<c_int>);
pub type CUresult = Result<(), CUerror>;

impl fmt::Debug for CUerror {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = self.0.get();
        let name_desc = CUerror::get_name_desc(code);
        if code == 218 || code == 200 {
            write!(
                f,
                "CUDA error: {:?} ({code}): {:?}  (executing `ptxas -arch sm_{{your gpu sm version}} your_ptx_code.ptx` might be helpful)",
                name_desc.0, name_desc.1,
            )
        } else {
            write!(
                f,
                "CUDA error: {:?} ({code}): {:?}",
                name_desc.0, name_desc.1
            )
        }
    }
}
#[path = "cuda_error/name_desc.rs"]
mod dumped;
#[cfg(feature = "native-error-desc")]
#[path = "cuda_error/dump_cudart_error.rs"]
mod dumper;

impl CUerror {
    const _ASSERT_SIZE_EQUAL: () = assert!(
        mem::size_of::<CUresult>() == mem::size_of::<c_int>(),
                                           "CUresult must be c_int"
    );
    /// get error code name and descriptions with cudart apis.
    #[cfg(feature = "native-error-desc")]
    pub unsafe fn cu_get_name_desc(code: c_int) -> (&'static CStr, &'static CStr) {
        unsafe {
            (
                dumper::cu_get_error_name(mem::transmute(code)),
                dumper::cu_get_error_string(mem::transmute(code)),
            )
        }
    }
    /// get error code name and descriptions with manualy exported code.
    pub fn get_name_desc(code: c_int) -> (&'static str, &'static str) {
        dumped::get_name_desc(code)
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUdevice(c_int);

/// CU context, should be dropped if it is not used anymore. Currently it is Device who creates and drops it.
/// If you have your own idea, do not forget drop it.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUcontext(*mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUmodule<'a>(*mut c_void, PhantomData<&'a ()>);
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUfunction<'a>(*mut c_void, PhantomData<&'a ()>);
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUstream(*mut c_void);
#[repr(transparent)]
pub struct PendingResult<'c>(PhantomData<&'c ()>);
// 手动绑定 CUDA 驱动 API
#[link(name = "cuda")]
unsafe extern "C" {
    #[must_use = "You should check whether the execution successes."]
    pub fn cuDeviceGetAttribute(result: &mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuInit(flags: c_uint) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuDeviceGetCount(count: &mut c_int) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    #[cfg_attr(feature = "using_v2_suffix", link_name = "cuCtxCreate_v2")]
    #[must_use = "You should check whether the execution successes."]
    pub fn cuCtxCreate(ctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuCtxSetLimit(limit: c_uint, size: usize) -> CUresult;
    #[cfg_attr(feature = "using_v2_suffix", link_name = "cuCtxDestroy_v2")]
    #[must_use = "You should check whether the execution successes."]
    pub fn cuCtxDestroy(ctx: CUcontext) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuModuleLoad(module: *mut CUmodule, ptx: *const c_char) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuModuleLoadData(module: *mut CUmodule, ptx: *const c_char) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    #[cfg_attr(feature = "using_v2_suffix", link_name = "cuMemAlloc_v2")]
    #[must_use = "You should check whether the execution successes."]
    pub fn cuMemAlloc(dptr: *mut *mut c_void, bytesize: usize) -> CUresult;
    #[cfg_attr(feature = "using_v2_suffix", link_name = "cuMemcpyHtoDAsync_v2")]
    #[must_use = "You should check whether the execution successes."]
    pub fn cuMemcpyHtoDAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult;
    #[cfg_attr(feature = "using_v2_suffix", link_name = "cuMemcpyDtoHAsync_v2")]
    #[must_use = "You should check whether the execution successes."]
    pub fn cuMemcpyDtoHAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult;
    // fn cuStreamCreate(stream: *mut CUstream, flag: c_uint) -> CUresult; // Parameters for stream creation (must be 0)
    #[must_use = "You should check whether the execution successes."]
    pub fn cuLaunchKernel(
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
    #[must_use = "You should check whether the execution successes."]
    pub fn cuFuncGetAttribute(result: &mut c_int, attrib: c_int, func: CUfunction) -> CUresult;
    #[must_use = "You should check whether the execution successes."]
    pub fn cuCtxSynchronize() -> CUresult; // not used yet.
    #[must_use = "You should check whether the execution successes."]
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
}

/// Cuda device and context
pub struct Device {
    #[allow(dead_code)]
    device: CUdevice,
    #[allow(dead_code)]
    context: CUcontext,
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe { cuCtxDestroy(self.context).unwrap() }
    }
}
impl Device {
    const STREAM: CUstream = CUstream(ptr::null_mut()); // default null stream
    /// the very fast approach to init first device and context, panic if init procedure contains errors.
    /// If you want to init more than the default GPU, use `init_all` instead.
    pub fn init() -> Self {
        let mut device = CUdevice(0);
        let mut ctx = CUcontext(ptr::null_mut());
        unsafe {
            cuInit(0).unwrap();
            cuDeviceGet(&mut device, 0).unwrap();
            cuCtxCreate(&mut ctx, 0, device).unwrap();
            cuCtxSetCurrent(ctx).unwrap();
            // cuCtxSetLimit(1, 1024 * 1024).unwrap();
        }
        Self {
            device,
            context: ctx,
        }
    }
    /// Init all GPUs. In case CUerror generates, return an error.
    pub fn init_all() -> Result<Vec<Self>, CUerror> {
        unsafe {
            cuInit(0)?; // Initialize the CUDA driver API Initializes the driver API and must be called before any other function from the driver API in the current process. Currently, the Flags parameter must be 0. If cuInit() has not been called, any function from the driver API will return CUDA_ERROR_NOT_INITIALIZED.
            let mut count = 0;
            cuDeviceGetCount(&mut count)?;
            let mut res = Vec::with_capacity(count as usize);
            for i in 0..count {
                let mut device = CUdevice(0);
                let mut ctx = CUcontext(ptr::null_mut());
                cuDeviceGet(&mut device, i)?;
                cuCtxCreate(&mut ctx, 0, device)?;
                cuCtxSetCurrent(ctx)?;
                // cuCtxSetLimit(1, 1024 * 1024)?;
                res.push(Self {
                    device,
                    context: ctx,
                });
            }
            Ok(res)
        }
    }
    /// Get major and minor CUDA capability version to calculate sm_** for generating better code.
    pub fn get_native_target_cpu_param(&self) -> Result<(c_int, c_int), CUerror> {
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            // According to https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
            cuDeviceGetAttribute(&mut major, 75, self.device)?;
            cuDeviceGetAttribute(&mut minor, 76, self.device)?;
        }
        Ok((major, minor))
    }
    /// Get equivlent of "-Ctarget-cpu=native" for this device
    /// panic if cuDeviceGetAttribute returns an error
    pub fn get_native_target_cpu(&self) -> Result<String, CUerror> {
        let (major, minor) = self.get_native_target_cpu_param()?;
        Ok(format!("-Ctarget-cpu=sm_{}{}", major, minor))
    }
    /// compile a module. Returns an error code 218 mostly means you do not send the correct PTX code into this function.
    #[must_use = "You should check whether the execution successes."]
    pub fn load<'a>(&'a self, file: &str) -> Result<CUmodule<'a>, CUerror> {
        if let Ok(cstr) = &CString::new(file) {
            self.load_raw(cstr)
        } else {
            Err(CUerror(NonZero::new(218).unwrap()))
        }
    }
    /// compile a module, with `&CStr` as its input
    #[must_use = "You should check whether the execution successes."]
    pub fn load_raw<'a>(&'a self, file: &CStr) -> Result<CUmodule<'a>, CUerror> {
        let mut module = CUmodule(ptr::null_mut(), PhantomData);
        unsafe { cuModuleLoad(&mut module, file.as_ptr() as _)? }
        Ok(module)
    }
    /// compile a module. Returns an error code 218 mostly means you do not send the correct PTX code into this function.
    #[must_use = "You should check whether the execution successes."]
    pub fn compile<'a>(&'a self, ptx: &str) -> Result<CUmodule<'a>, CUerror> {
        if let Ok(cstr) = &CString::new(ptx) {
            self.compile_raw(cstr)
        } else {
            Err(CUerror(NonZero::new(218).unwrap()))
        }
    }
    /// compile a module, with `&CStr` as its input
    #[must_use = "You should check whether the execution successes."]
    pub fn compile_raw<'a>(&'a self, c_ptx: &CStr) -> Result<CUmodule<'a>, CUerror> {
        let mut module = CUmodule(ptr::null_mut(), PhantomData);
        unsafe { cuModuleLoadData(&mut module, c_ptx.as_ptr() as _)? }
        Ok(module)
    }
    #[must_use = "You should check whether the execution successes."]
    pub fn set_print_buffer(size: usize) -> CUresult {
        unsafe { cuCtxSetLimit(1, size) }
    }
}
impl<'a> CUmodule<'a> {
    /// Get `CUfunction` from a module.
    #[must_use = "You should check whether the execution successes."]
    pub fn get_function<'b>(self, ptx: &str) -> Result<CUfunction<'b>, CUerror>
    where
        'a: 'b,
    {
        if let Ok(cstr) = &CString::new(ptx) {
            self.get_function_raw(cstr)
        } else {
            Err(CUerror(NonZero::new(218).unwrap()))
        }
    }
    /// Get `CUfunction` from a module, with `&CStr` as its input.
    #[must_use = "You should check whether the execution successes."]
    pub fn get_function_raw<'b>(self, function_name: &CStr) -> Result<CUfunction<'b>, CUerror>
    where
        'a: 'b,
    {
        let mut function = CUfunction(ptr::null_mut(), PhantomData);
        unsafe { cuModuleGetFunction(&mut function, self, function_name.as_ptr())? }
        Ok(function)
    }
}

impl<'b> CUfunction<'b> {
    /// Get major and minor CUDA capability version to calculate sm_** for generating better code.
    pub fn get_max_thread_per_block(&self) -> Result<c_int, CUerror> {
        let mut max_thread = 0;
        unsafe {
            // According to https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
            // CU_FUNC_ATTRIBUTE_MAX_THREAD_PER_BLOCK = 0
            cuFuncGetAttribute(&mut max_thread, 0, *self)?;
        }
        Ok(max_thread)
    }
    /// Call a CUfunction, take care!
    /// SAFETY: You should check very careful since it is a ffi call, and it calls an unsafe function.
    /// You should notice that, this is not marked as unsafe, but you should always remember, this is not a safe function.
    #[must_use = "You should check whether the execution successes."]
    pub fn call<'c, R>(self, param: Param<'c, R>) -> Result<PendingResult<'c>, CUerror>
    where
        'b: 'c,
    {
        // SAFETY: Massive ffi calls.
        unsafe {
            let len = param.len;
            if len == 0 {
                // SAFETY in NonZero::new_unchecked: 1 != 0
                Err(CUerror(NonZero::new_unchecked(1)))?
            }
            let length = param.result.len() * mem::size_of::<R>();
            let mut ret = ptr::null_mut();
            cuMemAlloc(&mut ret, length)?;
            cuMemcpyHtoDAsync(ret, param.result.as_ptr() as _, length, Device::STREAM)?;
            let mut device_mem = param
                .input
                .iter()
                .map(|_| ptr::null_mut())
                .collect::<Vec<_>>();
            for (device, &(ptr, size)) in device_mem.iter_mut().zip(param.input.iter()) {
                if size > 0 {
                    cuMemAlloc(device, size)?;
                    cuMemcpyHtoDAsync(*device, ptr, size, Device::STREAM)?
                }
            }
            let mut device_ref = device_mem
                .iter_mut()
                .chain(iter::once(&mut ret))
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
                ptr::null_mut(),
            )?;
            cuMemcpyDtoHAsync(param.result.as_mut_ptr() as _, ret, length, Device::STREAM)?
        }
        Ok(PendingResult(PhantomData))
    }
}

impl<'c> PendingResult<'c> {
    /// Wait for all the code finishes.
    #[must_use = "You should check whether the execution successes."]
    pub fn sync(self) -> CUresult {
        unsafe { cuStreamSynchronize(Device::STREAM) }
    }
}
