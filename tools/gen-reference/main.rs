//! `gen-reference`: regenerates or verifies `reference.json`.
//!
//! Recomputes every digest and vector in
//! [`pith_jpeg::reference`](../../src/reference) from the live
//! pipeline and either writes the repo-root file (`gen`) or verifies
//! the committed copy per-value and per-policy (`verify`, the CI
//! gate). Verification is semantic, never a whole-file byte compare:
//! the orthonormal-oracle vector's bits are platform-`libm`-shaped by
//! design and compared within the recorded tolerance instead.
//!
//! Usage:
//!
//! ```text
//! cargo run --locked --bin gen-reference -- gen   [path]   # default <manifest>/reference.json
//! cargo run --locked --bin gen-reference -- verify [path]  # default <manifest>/reference.json
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    let second = args.next();
    run(first.as_deref(), second)
}

/// The CLI body: `gen` writes a fresh render to `path`, `verify`
/// checks the committed copy per-value and per-policy.
fn run(mode: Option<&str>, path: Option<String>) -> ExitCode {
    let mode = match mode {
        Some(m @ ("gen" | "verify")) => m,
        _ => {
            eprintln!("usage: gen-reference <gen|verify> [path]");
            return ExitCode::from(2);
        }
    };
    let path = path.map_or_else(
        || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference.json"),
        PathBuf::from,
    );

    match mode {
        "gen" => match std::fs::write(&path, pith_jpeg::reference::reference_json()) {
            Ok(()) => {
                println!("wrote {}", path.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("FAIL: writing {}: {e}", path.display());
                ExitCode::FAILURE
            }
        },
        _ => match std::fs::read_to_string(&path) {
            Ok(committed) => match pith_jpeg::reference::verify_str(&committed) {
                Ok(()) => {
                    println!("reference vectors are current");
                    ExitCode::SUCCESS
                }
                Err(why) => {
                    eprintln!("FAIL: {why}");
                    ExitCode::FAILURE
                }
            },
            Err(e) => {
                eprintln!("FAIL: cannot read {}: {e}", path.display());
                ExitCode::FAILURE
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{ExitCode, run};

    /// gen -> verify round-trips through an explicit path; a missing
    /// file and a bad mode fail with the documented codes.
    #[test]
    fn gen_verify_roundtrip_and_failures() {
        let path = std::env::temp_dir().join(format!(
            "pith-jpeg-genref-{}-{}.json",
            std::process::id(),
            line!()
        ));
        let p = path.to_string_lossy().into_owned();

        assert_eq!(run(Some("gen"), Some(p.clone())), ExitCode::SUCCESS);
        assert_eq!(run(Some("verify"), Some(p.clone())), ExitCode::SUCCESS);

        let absent = format!("{p}.absent");
        assert_eq!(run(Some("verify"), Some(absent)), ExitCode::FAILURE);

        std::fs::write(&path, "{}").expect("write broken file");
        assert_eq!(run(Some("verify"), Some(p)), ExitCode::FAILURE);

        let _ = std::fs::remove_file(&path);
    }

    /// Unknown modes are a usage error with exit code 2.
    #[test]
    fn usage_errors() {
        assert_eq!(run(None, None), ExitCode::from(2));
        assert_eq!(run(Some("bogus"), None), ExitCode::from(2));
    }

    /// The binary links the library's vector table.
    #[test]
    fn bin_links_the_library() {
        assert!(!pith_jpeg::reference::vectors().is_empty());
    }
}
