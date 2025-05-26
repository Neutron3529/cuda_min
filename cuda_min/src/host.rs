#[path = "cuda.rs"]
mod cuda;
pub use cuda::*;
// #[Deprecated(since = "0.1.4", note = "Device must be stored, use `let device = Device::init(); let {result} = device.compile({your input})` instead.")]
// pub fn compile(s: &str) -> Result<CUmodule, CUerror> {
//     Device::init().compile(s)
// }
// #[Deprecated(since = "0.1.4", note = "Device must be stored, use `let device = Device::init(); let {result} = device.load({your input})` instead.")]
// pub fn load(s: &str) -> Result<CUmodule, CUerror> {
//     Device::init().load(s)
// }
/// Function Param
///
/// Packing and transfering parameters and returns to and from GPU.
#[derive(Debug)]
pub struct Param<'a, R> {
    /// input pointers and lengths (in bytes)
    pub input: Vec<(*const core::ffi::c_void, usize)>,
    /// result, controls the numbers of tasks
    pub result: &'a mut [R],
    /// shared memory size, I have not yet tested it.
    pub shared_mem: u32,
    block_size: (u32, u32, u32),
    grid_size: (u32, u32, u32),
    len: usize,
}
impl<'a, R> Param<'a, R> {
    /// Set block 1d size. Currently only 1d size could be set directly (since result is a 1d vector)
    pub fn block_size(mut self, val: u32) -> Self {
        self.block_size = (val, 1, 1);
        self.grid_size = ((self.len / self.block_size.0 as usize) as u32, 1, 1);
        self
    }
    /// Set block size and grid size. You must ensure you handled it very well.
    /// Although all the functions are not very safe. This function is extremely unsafe.
    pub unsafe fn set_block_grid_size(
        mut self,
        block_size: (u32, u32, u32),
        grid_size: (u32, u32, u32),
    ) -> Self {
        self.block_size = block_size;
        self.grid_size = grid_size;
        self.len = block_size.0 as usize * grid_size.0 as usize;
        self
    }
    /// Generate parameter from its output, use output's length as number of tasks.
    pub fn new(result: &'a mut [R]) -> Self {
        let len = result.len();
        let block_size = (32.min(len as u32), 1, 1);
        let grid_size = ((len / block_size.0 as usize) as u32, 1, 1);
        Self {
            input: Vec::new(),
            result,
            shared_mem: 0,
            block_size,
            grid_size,
            len,
        }
    }
    /// Convenience push method, panic if the length is incorrect.
    pub fn push<T>(self, item: &[T]) -> Self {
        self.checked_push(item).unwrap_or_else(|x| x)
    }
    /// Set real length
    pub fn len(mut self, len: usize) -> Self {
        self.len = len;
        self.grid_size.0 = len as u32 / self.block_size.0;
        self
    }
    /// Push vectors into this parameter collection.
    pub fn checked_push<T>(mut self, item: &[T]) -> Result<Self, Self> {
        // let size = core::mem::size_of_val(item);
        if let Some(size) = core::num::NonZero::new(core::mem::size_of::<T>() * item.len()) {
            self.input.push((item.as_ptr() as _, size.get()));
            Ok(self)
        } else {
            self.input.push((core::ptr::null(), 0));
            Err(self)
        }
    }
    /// Set shared mem
    pub fn shared(mut self, size: u32) -> Self {
        self.shared_mem = size;
        self
    }
}
