//! Graffiti Web Interface
//!
//! WebAssembly-powered web interface for the revolutionary proof-based search engine

use wasm_bindgen::prelude::*;
use graffiti_core::*;

// WebAssembly entry point
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    
    #[cfg(feature = "wee_alloc")]
    {
        static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
        #[global_allocator]
        static GLOBAL: &wee_alloc::WeeAlloc = &ALLOC;
    }
}

/// Web server for HTTP interface (non-WASM)
#[cfg(not(target_arch = "wasm32"))]
pub struct WebServer {
    engine: std::sync::Arc<crate::GraffitiSearchEngine>,
}

#[cfg(not(target_arch = "wasm32"))]
impl WebServer {
    pub fn new(engine: std::sync::Arc<crate::GraffitiSearchEngine>) -> Self {
        Self { engine }
    }
    
    pub async fn start(&self, port: u16) -> GraffitiResult<()> {
        // TODO: Implement web server
        println!("Web server starting on port {}", port);
        Ok(())
    }
}

// WASM bindings for browser interface
#[wasm_bindgen]
pub struct GraffitiWeb {
    // TODO: Add WASM-specific state
}

#[wasm_bindgen]
impl GraffitiWeb {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }
    
    #[wasm_bindgen]
    pub async fn search(&self, query: &str) -> String {
        // TODO: Implement WASM search interface
        format!("Searching for: {}", query)
    }
}
