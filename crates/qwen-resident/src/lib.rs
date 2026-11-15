//! Resident Qwen serving with an isolated, runtime-loaded llama.cpp library.
pub mod contract;
#[cfg(all(feature = "native-qwen", windows))]
mod native;
#[cfg(feature = "native-qwen")]
pub mod serving;
