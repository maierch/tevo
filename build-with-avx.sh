#!/bin/bash
RUSTFLAGS='-C target-feature=+sse3,+avx' cargo build --release