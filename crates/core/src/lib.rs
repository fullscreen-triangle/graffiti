//! Graffiti Search Core
//!
//! Core types and traits for the revolutionary proof-based search engine that constructs
//! mathematical proofs from environmental reality using atmospheric molecular processing,
//! temporal coordination, and strategic impossibility optimization.

pub mod types;
pub mod traits;
pub mod error;
pub mod constants;

// Revolutionary algorithm integrations
pub mod honjo_integration;
pub mod self_aware_integration;  
pub mod kinshasa_integration;

// Re-export core types for convenience
pub use types::*;
pub use traits::*;
pub use error::*;
pub use constants::*;
