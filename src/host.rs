#[path = "cuda.rs"]
mod cuda;
pub use cuda::Device;
use cuda::*;
pub fn compile(s: &str) -> Result<CUmodule, CUerror> {
    Device::init().compile(s)
}
pub fn load(s: &str) -> Result<CUmodule, CUerror> {
    Device::init().load(s)
}
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
}
impl<'a, R> Param<'a, R> {
    /// Set block 1d size. Currently only 1d size could be set directly (since result is a 1d vector)
    pub fn block_size(&mut self, val: u32) {
        self.block_size = (val, 1, 1);
        self.grid_size = (
            (self.result.len() / self.block_size.0 as usize) as u32,
                          1,
                          1,
        );
    }
    /// Set block size and grid size. You must ensure you handled it very well.
    /// Although all the functions are not very safe. This function is extremely unsafe.
    pub unsafe fn set_block_grid_size(
        &mut self,
        block_size: (u32, u32, u32),
                                      grid_size: (u32, u32, u32),
    ) {
        self.block_size = block_size;
        self.grid_size = grid_size;
    }
    /// generate parameter from its output, use output's length as number of tasks.
    pub fn new(result: &'a mut [R]) -> Self {
        let mut block_size = (
            32.max(((result.len() >> 32).max(1)).ilog(2) + 1) as u32,
                              1,
                              1,
        );
        if result.len() <= u32::MAX as usize {
            block_size.0 = (result.len() as u32).min(block_size.0);
        }
        let grid_size = ((result.len() / block_size.0 as usize) as u32, 1, 1);
        Self {
            input: Vec::new(),
            result,
            shared_mem: 0,
            block_size,
            grid_size,
        }
    }
    /// push vectors into this parameter collection.
    pub fn push<T>(&mut self, item: &[T]) -> bool {
        // let size = core::mem::size_of_val(item);
        if let Some(size) = core::num::NonZero::new(core::mem::size_of::<T>() * item.len()) {
            self.input.push((item.as_ptr() as _, size.get()));
            true
        } else {
            false
        }
    }
}
