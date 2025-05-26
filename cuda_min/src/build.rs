pub struct GpuCode<'a, 'b, 'c, 'd> {
    pub gpu_crate_name: &'a str,
    pub gpu_crate_dir: &'b str,
    /// you should set it into something diffrerent from `debug` or `release` to avoid deadlock.
    pub profile_name: &'c str,
    /// set CARGO_TARGET_DIR or CARGO_BUILD_TARGET_DIR might be better.
    pub fallback_target_dir: &'d str,
    /// Indicate whether the whole target folder should be removed after a successful build.
    pub clean: bool,
}
impl<'a, 'b, 'c, 'd> GpuCode<'a, 'b, 'c, 'd> {
    pub fn new(gpu_crate_name: &'a str, gpu_crate_dir: &'b str) -> Self {
        Self {
            gpu_crate_name,
            gpu_crate_dir,
            profile_name: "cuda",
            fallback_target_dir: "target",
            clean: false,
        }
    }
    pub fn profile(mut self, profile_name: &'c str) -> Self {
        self.profile_name = profile_name;
        self
    }
    pub fn target(mut self, fallback_target_dir: &'d str) -> Self {
        self.fallback_target_dir = fallback_target_dir;
        self
    }
    /// Will remove the whole target folder after a successful build.
    /// PLEASE ENSURE you specific a safe folder (e.g., execute `.target(env!("OUT_DIR"))`) otherwise important files (e.g., your compiled cuda program) will be removed.
    pub fn clean(mut self) -> Self {
        self.clean = true;
        self
    }
    pub fn build(self) {
        build(
            self.gpu_crate_name,
            self.gpu_crate_dir,
            self.fallback_target_dir,
            self.profile_name,
            self.clean,
        )
    }
}

fn build(name: &str, nvptx_dir: &str, target_dir: &str, profile: &str, clean: bool) {
    use core::fmt::Display;
    use std::{env, process::Command};

    fn warning(x: impl Display) {
        println!(
            "{}",
            format!("cargo::warning={x}").replace("\n", "\ncargo::warning=")
        )
    }

    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo::rerun-if-changed={nvptx_dir}/src");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=CUDA_PATH");

    if let Ok(cuda_path) = env::var("CUDA_PATH").or(env::var("CUDA_HOME")) {
        warning(format_args!("try searching with cudart since CUDA_PATH or CUDA_HOME are not all None.\nprint rustc-link-search={cuda_path}/lib"));
        println!("cargo:rustc-link-search={cuda_path}/lib");
    } else {
        warning(
            "Neither environment CUDA_PATH nor CUDA_HOME exists, feature `cudart` might failed to compile. Please specific one of $CUDA_PATH or $CUDA_HOME",
        )
    }

    let cargo = env::var("CARGO");
    let cargo = cargo.as_deref().unwrap_or("cargo");
    if let Ok(rustflags) = env::var("CARGO_ENCODED_RUSTFLAGS") {
        warning(format_args!(
            "omit CARGO_ENCODED_RUSTFLAGS={} for compiling GPU code.",
            rustflags.replace(0x1f as char, " ")
        ))
    }
    if let Ok(rustflags) = env::var("RUSTFLAGS") {
        warning(format_args!(
            r#"omit RUSTFLAGS={rustflags} for compiling GPU code.
  In case you really need such code, adding something like:
  ```
  [target.'cfg(target_arch="nvptx64")']
  rustflags = ["-Ctarget-cpu=sm_120", "-Clinker=llvm-bitcode-linker"]
  ```
  In `.cargo/config.toml` where GPU code could touch.
"#
        ))
    }
    let target = env::var("CARGO_TARGET_DIR").or(env::var("CARGO_BUILD_TARGET_DIR"));
    let target = target.as_deref().unwrap_or(target_dir);
    let out = env::var("OUT_DIR").unwrap();
    warning(format_args!(
        "executing\x1b[1;32m \"{cargo}\" build --color always --profile {profile} --manifest-path \"{nvptx_dir}/Cargo.toml\" --target nvptx-nvidia-cuda --target-dir \"{target}\" \x1b[m"
    ));
    match Command::new(cargo)
        .args([
            "build",
            "--color",
            "always",
            "--profile",
            profile,
            "--manifest-path",
            &format!("{nvptx_dir}/Cargo.toml"),
            "--target",
            "nvptx64-nvidia-cuda",
            "--target-dir",
            &target,
        ])
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS") // otherwise some unnecessary code such as "-Ctarget-cpu=native" will be sent to the compiler
        .output()
    {
        Err(e) => {
            panic!(
                "executing\x1b[1;32m \"{cargo}\" build --color always --profile {profile} --manifest-path \"{nvptx_dir}/Cargo.toml\" --target nvptx-nvidia-cuda --target-dir \"{target}\" \x1b[mfailed: {e:?}"
            )
        }
        Ok(result) => {
            if !result.status.success() {
                panic!(
                    "compile failed:\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&result.stdout),
                    String::from_utf8_lossy(&result.stderr)
                )
            } else {
                if result.stderr.len() > 0 {
                    warning(String::from_utf8_lossy(&result.stderr))
                }
                let mut pb = std::path::PathBuf::new();
                pb.push(out);
                pb.push(&format!("{name}.ptx"));
                if pb.exists() {
                    std::fs::remove_file(&pb).unwrap()
                }
                warning(format_args!(
                    "linking {target}/nvptx64-nvidia-cuda/cuda/{name}.ptx to {pb:?}, current_dir = {:?}",
                    env::current_dir()
                ));
                std::fs::hard_link(
                    &format!("{target}/nvptx64-nvidia-cuda/cuda/{name}.ptx"),
                    &pb,
                )
                .unwrap();
                if clean {
                    std::fs::remove_dir_all(&target)
                        .unwrap_or_else(|e| warning(format_args!("Auto clean failed: {e:?}")));
                }
            }
        }
    }
}
