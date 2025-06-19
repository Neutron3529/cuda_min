# cuda_min

This crate is mainly created for providing a simple method for utilizing the nvptx64 backend.

## Usage

```
# Prepare
rustup toolchain install nightly # nightly version is required to enable nvptx64 backend.
## Optional
rustup component add llvm-tools llvm-bitcode-linker # cuda_min's `build-script-with-llvm-bitcode-linker` feature requires `llvm-tools` and `llvm-bitcode-linker`.

# Create crates and add dependencies

PWD="$(pwd)" # change it to your folder in case you want.
cargo new cpu_code # your main crate
cargo new gpu_code # your gpu crate

# Add dependencies
cd $PWD/cpu_code
cargo add cuda_min
cd $PWD/gpu_code
cargo add cuda_min # backend might need an `abort` function, or a `panic_handler`.

# Optional: configure build script, has 2 steps.

## Step 1. CPU side.
cd $PWD/cpu_code
cargo add cuda_min --build # add convenient build support
cat > build.rs <<EOF
fn main() {
    cuda_min::GpuCode::new("gpu_code", "../gpu_code")
        // .target(env!("OUT_DIR")) // specific the build target (relative to `build.rs`), default is "target". In a workspace, you might want to specific this target rather than using the default output dir. 
        // .profile("debug") // Not recommended.
        // .clean() // Will remove the whole target folder, PLEASE ENSURE you specific a safe folder (e.g., execute `.target(env!("OUT_DIR"))`) otherwise important files (e.g., your compiled cuda program) will be removed.
        .build()
}
EOF

## Step 2. GPU side.
cd $PWD/gpu_code
cat >> Cargo.toml <<EOF
[profile.cuda] # avoid lock
inherits = "release"
EOF

# Edit your code 
# ...

# Compile
cargo build --release # build.rs will generate a `{gpu_crate_name}.ptx` in folder `env!("OUT_DIR")`.
# In this case, you could directly obtain the PTX file using `const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "gpu_code.ptx"))`
```
## Features

### default

["panic-handler", "build-script-with-llvm-bitcode-linker", "using_v2_suffix"]

### panic-handler

Affect only GPU side.

Adding a default panic-handler (abort) to make rustc happy. Disable it in case you have another panic-handler.

### build-script-with-llvm-bitcode-linker 

Affect only CPU side.

Won't yield error unless you create a `cuda_min::GpuCode` struct and call `.build()` accidently.

### cudart

Affect only CPU side.

You must make sure `$CUDA_PATH/lib` in  `{LD,}_LIBRARY_PATH` thus cargo can find the cudart library. Currently a `cuda_error_code_generator.rs` script is used for generating cuda error related scripts.

## Examples

There are various examples in https://github.com/Neutron3529/cuda_min
