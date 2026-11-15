//! Native Vulkan compute and bounded target-only text decoding for Strix Halo.
#[cfg(all(feature = "native-aten", windows))]
pub mod aten;
pub mod checkpoint;
pub mod config;
pub mod decoder;
pub mod engram;
pub mod math;
pub mod reference;
pub mod runtime;
pub mod verification;
pub mod weights;
