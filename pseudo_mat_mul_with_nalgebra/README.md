# Example - mat_mul

This folder shows a simple way for Executing mat_mul, where a mat is defined in a 3rd-party (nalgebra)

## * Using `build.rs`

The steps are very simple, just execute `cargo build` is enough. The build script will do all the remain jobs for you.

In this way, the `.cargo` folder in gpu_code is not necessary, since `build.rs` told `cargo` what the *target-triple* is

## * Manual build

In case `build.rs` do not fit your needs, you could compile manually:

```
cargo build --color always --profile cuda --manifest-path "gpu_code/Cargo.toml" --target nvptx-nvidia-cuda --target-dir "target"
```

You could just execute what `build.rs` has executed, or make a `.cargo/config.toml` file in gpu_code's folder:

```toml
[build]
target = "nvptx64-nvidia-cuda"

[rust-analyzer]
cargo.target = "nvptx64-nvidia-cuda"
```

