$env:RUSTFLAGS="-C target-feature=+sse3,+avx,+avx2"
cargo build --release