//! Build script: compiles the hand-written assembly microkernels.
//!
//! Sets `vearo_cpu_asm` only on `x86_64`, so other architectures fall back to
//! the portable Rust path rather than failing to build.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(vearo_cpu_asm)");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if target_arch == "x86_64" {
        cc::Build::new()
            .file("src/asm/x86_64/matmul.s")
            .compile("vearo_cpu_asm");
        println!("cargo:rustc-cfg=vearo_cpu_asm");
    }
    println!("cargo:rerun-if-changed=src/asm");
}
