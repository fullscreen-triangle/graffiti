//! Constants for the Graffiti Search revolutionary proof-based search engine

use std::time::Duration;

/// Physical and mathematical constants
pub mod physics {
    /// Earth's total atmospheric molecules (approximately 10^44)
    pub const ATMOSPHERIC_MOLECULES: u64 = 10_000_000_000_000_000_000_000_000_000_000_000_000_000_000_000;
    
    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
    
    /// Planck constant (J⋅s)
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;
    
    /// Boltzmann constant (J/K)
    pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;
    
    /// Avogadro constant (mol^-1)
    pub const AVOGADRO_NUMBER: f64 = 6.02214076e23;
    
    /// Universal gas constant (J⋅mol^-1⋅K^-1)
    pub const GAS_CONSTANT: f64 = 8.314462618;
    
    /// Atmospheric pressure at sea level (Pa)
    pub const ATMOSPHERIC_PRESSURE_SEA_LEVEL: f64 = 101_325.0;
    
    /// Earth's gravitational acceleration (m/s²)
    pub const EARTH_GRAVITY: f64 = 9.80665;
}

/// Environmental measurement constants
pub mod environmental {
    /// Number of fundamental environmental dimensions
    pub const DIMENSIONS: usize = 12;
    
    /// Environmental dimension names
    pub const DIMENSION_NAMES: [&str; 12] = [
        "biometric", "spatial", "atmospheric", "cosmic",
        "temporal", "hydrodynamic", "geological", "quantum",
        "computational", "acoustic", "ultrasonic", "visual"
    ];
    
    /// Default measurement precision (relative error)
    pub const DEFAULT_PRECISION: f64 = 1e-6;
    
    /// Minimum stable measurement duration
    pub const MIN_MEASUREMENT_DURATION: std::time::Duration = std::time::Duration::from_millis(100);
    
    /// Maximum environmental state age before refresh
    pub const MAX_STATE_AGE: std::time::Duration = std::time::Duration::from_secs(1);
    
    /// Environmental uniqueness threshold
    pub const UNIQUENESS_THRESHOLD: f64 = 0.999;
}

/// Atmospheric molecular processing constants
pub mod atmospheric {
    /// Primary atmospheric constituents
    pub const N2_PERCENTAGE: f64 = 78.084;
    pub const O2_PERCENTAGE: f64 = 20.946;
    pub const AR_PERCENTAGE: f64 = 0.9340;
    pub const CO2_PERCENTAGE: f64 = 0.0413;
    pub const H2O_MAX_PERCENTAGE: f64 = 4.0;
    
    /// Molecular processing parameters
    pub const MIN_PROCESSORS_ACTIVE: u64 = 1_000_000;
    pub const TARGET_PROCESSORS_ACTIVE: u64 = 1_000_000_000;
    pub const MAX_PROCESSORS_ACTIVE: u64 = physics::ATMOSPHERIC_MOLECULES;
    
    /// Network consensus thresholds
    pub const CONSENSUS_THRESHOLD: f64 = 0.667; // 2/3 majority
    pub const STRONG_CONSENSUS_THRESHOLD: f64 = 0.9;
    pub const WEAK_CONSENSUS_THRESHOLD: f64 = 0.51;
    
    /// Molecular oscillation frequencies (Hz)
    pub const N2_VIBRATIONAL_FREQUENCY: f64 = 2.36e14;
    pub const O2_VIBRATIONAL_FREQUENCY: f64 = 4.74e14;
    pub const H2O_ROTATIONAL_FREQUENCY_RANGE: (f64, f64) = (1e11, 1e12);
}

/// S-entropy strategic impossibility constants
pub mod s_entropy {
    /// S-entropy dimension weights
    pub const S_KNOWLEDGE_WEIGHT: f64 = 0.33;
    pub const S_TIME_WEIGHT: f64 = 0.33;
    pub const S_ENTROPY_WEIGHT: f64 = 0.34;
    
    /// Strategic impossibility thresholds
    pub const IMPOSSIBILITY_THRESHOLD: f64 = f64::INFINITY;
    pub const STRATEGIC_WINDOW_THRESHOLD: f64 = 1e6;
    
    /// Alternating weight factors for impossibility cancellation
    pub const ALPHA_BASE: f64 = 1.0;
    pub const BETA_BASE: f64 = 2.0;
    pub const GAMMA_BASE: f64 = 1.414; // √2
    
    /// Cross-product interaction strengths
    pub const LAMBDA_12: f64 = 0.1;
    pub const LAMBDA_13: f64 = 0.15;
    pub const LAMBDA_23: f64 = 0.2;
}

/// Temporal coordination constants (Sango Rine Shumba)
pub mod temporal {
    use std::time::Duration;
    
    /// Target temporal precision (seconds)
    pub const TARGET_PRECISION: f64 = 1e-30;
    
    /// Precision-by-difference enhancement factor
    pub const PRECISION_ENHANCEMENT_FACTOR: f64 = 10.0;
    
    /// Temporal fragment coordination windows
    pub const MIN_COHERENCE_WINDOW: Duration = Duration::from_millis(1);
    pub const MAX_COHERENCE_WINDOW: Duration = Duration::from_secs(1);
    pub const DEFAULT_COHERENCE_WINDOW: Duration = Duration::from_millis(100);
    
    /// Temporal safety margins
    pub const SAFETY_MARGIN: Duration = Duration::from_micros(10);
    pub const PROCESSING_BUFFER: Duration = Duration::from_millis(1);
    
    /// Fragment coordination limits
    pub const MAX_FRAGMENTS_PER_MESSAGE: usize = 1000;
    pub const MIN_FRAGMENTS_PER_MESSAGE: usize = 1;
    pub const DEFAULT_FRAGMENTS_PER_MESSAGE: usize = 8;
}

/// BMD (Biological Maxwell Demon) processing constants
pub mod bmd {
    /// Frame selection parameters
    pub const MAX_FRAMES_PER_QUERY: usize = 100;
    pub const MIN_FRAMES_PER_QUERY: usize = 1;
    pub const DEFAULT_FRAMES_PER_QUERY: usize = 10;
    
    /// Frame weight thresholds
    pub const MIN_FRAME_WEIGHT: f64 = 0.01;
    pub const MAX_FRAME_WEIGHT: f64 = 1.0;
    pub const DEFAULT_FRAME_WEIGHT: f64 = 0.5;
    
    /// Selection probability parameters
    pub const SELECTION_TEMPERATURE: f64 = 1.0;
    pub const SELECTION_DECAY_FACTOR: f64 = 0.99;
    
    /// Memory frame cache limits
    pub const MAX_CACHED_FRAMES: usize = 10_000;
    pub const FRAME_CACHE_TTL_SECONDS: u64 = 3600; // 1 hour
}

/// Proof construction constants
pub mod proof {
    /// Proof validation thresholds
    pub const MIN_PROOF_CONFIDENCE: f64 = 0.5;
    pub const HIGH_PROOF_CONFIDENCE: f64 = 0.8;
    pub const VERY_HIGH_PROOF_CONFIDENCE: f64 = 0.95;
    
    /// Construction method priorities
    pub const ENVIRONMENTAL_PRIORITY: u8 = 1;
    pub const ATMOSPHERIC_PRIORITY: u8 = 2;
    pub const S_ENTROPY_PRIORITY: u8 = 3;
    pub const TEMPORAL_PRIORITY: u8 = 4;
    pub const BMD_PRIORITY: u8 = 5;
    pub const THERMODYNAMIC_PRIORITY: u8 = 6;
    
    /// Proof step limits
    pub const MAX_PROOF_STEPS: usize = 1000;
    pub const MIN_PROOF_STEPS: usize = 1;
    pub const TYPICAL_PROOF_STEPS: usize = 10;
}

/// Perturbation validation constants
pub mod perturbation {
    /// Stability thresholds
    pub const MIN_STABILITY_SCORE: f64 = 0.5;
    pub const GOOD_STABILITY_SCORE: f64 = 0.8;
    pub const EXCELLENT_STABILITY_SCORE: f64 = 0.95;
    
    /// Perturbation test parameters
    pub const WORD_REMOVAL_TESTS: usize = 10;
    pub const REARRANGEMENT_TESTS: usize = 5;
    pub const SUBSTITUTION_TESTS: usize = 8;
    pub const NEGATION_TESTS: usize = 3;
    
    /// Perturbation intensity levels
    pub const LIGHT_PERTURBATION: f64 = 0.1;
    pub const MODERATE_PERTURBATION: f64 = 0.3;
    pub const STRONG_PERTURBATION: f64 = 0.7;
}

/// Performance and resource constants
pub mod performance {
    use std::time::Duration;
    
    /// Response time targets
    pub const TARGET_RESPONSE_TIME: Duration = Duration::from_millis(100);
    pub const MAX_ACCEPTABLE_RESPONSE_TIME: Duration = Duration::from_secs(1);
    pub const ZERO_LATENCY_THRESHOLD: Duration = Duration::from_millis(10);
    
    /// Resource limits
    pub const MAX_CONCURRENT_QUERIES: usize = 1000;
    pub const MAX_MEMORY_USAGE_MB: usize = 2048;
    pub const MAX_CPU_USAGE_PERCENT: f64 = 80.0;
    
    /// Cache settings
    pub const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(300); // 5 minutes
    pub const MAX_CACHE_SIZE_MB: usize = 512;
    pub const CACHE_EVICTION_THRESHOLD: f64 = 0.8;
}

/// Network and communication constants
pub mod network {
    /// Default ports
    pub const DEFAULT_HTTP_PORT: u16 = 8080;
    pub const DEFAULT_HTTPS_PORT: u16 = 8443;
    pub const DEFAULT_METRICS_PORT: u16 = 9090;
    
    /// Timeouts
    pub const DEFAULT_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    pub const KEEP_ALIVE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
    pub const CONNECTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    
    /// Rate limiting
    pub const DEFAULT_RATE_LIMIT: u32 = 100; // requests per minute
    pub const BURST_LIMIT: u32 = 20;
}

/// Error handling constants
pub mod errors {
    /// Retry configuration
    pub const MAX_RETRY_ATTEMPTS: u32 = 3;
    pub const RETRY_BACKOFF_MS: u64 = 100;
    pub const RETRY_BACKOFF_MULTIPLIER: f64 = 2.0;
    
    /// Circuit breaker configuration
    pub const CIRCUIT_BREAKER_FAILURE_THRESHOLD: u32 = 5;
    pub const CIRCUIT_BREAKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    pub const CIRCUIT_BREAKER_SUCCESS_THRESHOLD: u32 = 2;
}

/// System limits and constraints
pub mod limits {
    /// Query processing limits
    pub const MAX_QUERY_LENGTH: usize = 10_000;
    pub const MIN_QUERY_LENGTH: usize = 1;
    
    /// Response size limits
    pub const MAX_RESPONSE_SIZE_MB: usize = 10;
    pub const MAX_PROOF_SIZE_KB: usize = 100;
    pub const MAX_EVIDENCE_ITEMS: usize = 1000;
    
    /// Platform limits
    pub const MAX_RESOLUTION_PLATFORMS_PER_QUERY: usize = 10;
    pub const MAX_POINTS_PER_RESOLUTION: usize = 100;
    pub const MAX_AFFIRMATIONS_PER_POINT: usize = 50;
    pub const MAX_CONTENTIONS_PER_POINT: usize = 50;
}

/// Version information
pub mod version {
    /// Current version of the Graffiti Search engine
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");
    
    /// Build information
    pub const BUILD_TIME: &str = env!("BUILD_TIME");
    pub const BUILD_COMMIT: &str = env!("BUILD_COMMIT");
    
    /// API version
    pub const API_VERSION: &str = "v1";
}

/// Default configuration values
pub mod defaults {
    use super::*;
    
    /// Default environmental measurement interval
    pub const MEASUREMENT_INTERVAL: Duration = Duration::from_millis(100);
    
    /// Default atmospheric processing batch size
    pub const ATMOSPHERIC_BATCH_SIZE: usize = 1000;
    
    /// Default temporal fragment size
    pub const TEMPORAL_FRAGMENT_SIZE: usize = 256;
    
    /// Default BMD frame cache size
    pub const BMD_FRAME_CACHE_SIZE: usize = 1000;
    
    /// Default proof construction timeout
    pub const PROOF_CONSTRUCTION_TIMEOUT: Duration = Duration::from_secs(30);
    
    /// Default perturbation test intensity
    pub const PERTURBATION_INTENSITY: f64 = perturbation::MODERATE_PERTURBATION;
}
