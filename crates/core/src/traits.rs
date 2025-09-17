//! Core traits for the Graffiti Search engine architecture

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

use crate::{
    EnvironmentalState, Query, SearchResponse, Point, Resolution, Proof,
    InformationMolecule, SEntropyCoordinates, TemporalFragment, BMDFrame,
    GraffitiResult, QueryId,
};

/// Core trait for environmental measurement across twelve dimensions
#[async_trait]
pub trait EnvironmentalMeasurement: Send + Sync {
    /// Measure current environmental state across all twelve dimensions
    async fn measure_environment(&self) -> GraffitiResult<EnvironmentalState>;

    /// Measure specific environmental dimension
    async fn measure_dimension(&self, dimension: &str) -> GraffitiResult<f64>;

    /// Get environmental measurement capabilities
    fn get_capabilities(&self) -> Vec<String>;

    /// Calibrate environmental sensors
    async fn calibrate(&mut self) -> GraffitiResult<()>;

    /// Check if environmental measurement is stable
    async fn is_stable(&self) -> GraffitiResult<bool>;
}

/// Trait for atmospheric molecular processing using Earth's 10^44 molecules
#[async_trait]
pub trait AtmosphericProcessing: Send + Sync {
    /// Process information using atmospheric molecular network
    async fn process_atmospheric(
        &self,
        information: Vec<InformationMolecule>,
    ) -> GraffitiResult<Vec<InformationMolecule>>;

    /// Get number of available atmospheric processors
    async fn get_processor_count(&self) -> GraffitiResult<u64>;

    /// Check atmospheric network health
    async fn network_health(&self) -> GraffitiResult<f64>;

    /// Coordinate molecular consensus for validation
    async fn coordinate_consensus(
        &self,
        molecules: &[InformationMolecule],
    ) -> GraffitiResult<f64>;
}

/// Trait for S-entropy strategic impossibility optimization
#[async_trait]
pub trait SEntropyNavigation: Send + Sync {
    /// Navigate to strategic impossibility coordinates
    async fn navigate_impossibility(
        &self,
        coordinates: SEntropyCoordinates,
    ) -> GraffitiResult<Vec<Proof>>;

    /// Calculate S-entropy coordinates for given problem
    async fn calculate_coordinates(&self, problem: &str) -> GraffitiResult<SEntropyCoordinates>;

    /// Check if coordinates represent strategic impossibility window
    async fn is_strategic_impossibility(&self, coordinates: &SEntropyCoordinates) -> GraffitiResult<bool>;

    /// Optimize strategic combination of impossible approaches
    async fn optimize_strategic_combination(
        &self,
        impossible_approaches: Vec<String>,
    ) -> GraffitiResult<String>;
}

/// Trait for temporal coordination (Sango Rine Shumba framework)
#[async_trait]
pub trait TemporalCoordination: Send + Sync {
    /// Fragment information for temporal coordination
    async fn fragment_information(
        &self,
        content: String,
        delivery_time: std::time::SystemTime,
    ) -> GraffitiResult<Vec<TemporalFragment>>;

    /// Coordinate temporal fragment delivery
    async fn coordinate_delivery(
        &self,
        fragments: Vec<TemporalFragment>,
    ) -> GraffitiResult<()>;

    /// Calculate precision-by-difference enhancement
    async fn calculate_precision_enhancement(&self) -> GraffitiResult<f64>;

    /// Predict optimal information delivery timing
    async fn predict_delivery_timing(&self, query: &Query) -> GraffitiResult<std::time::SystemTime>;
}

/// Trait for BMD (Biological Maxwell Demon) processing
#[async_trait]
pub trait BMDProcessing: Send + Sync {
    /// Select appropriate frames for query processing
    async fn select_frames(&self, query: &Query) -> GraffitiResult<Vec<BMDFrame>>;

    /// Fuse experience with memory frames
    async fn fuse_experience_frame(
        &self,
        experience: &EnvironmentalState,
        frames: &[BMDFrame],
    ) -> GraffitiResult<Point>;

    /// Validate sanity through collective consensus
    async fn validate_sanity(&self, point: &Point) -> GraffitiResult<f64>;

    /// Update frame weights based on usage
    async fn update_frame_weights(&mut self, frame_ids: &[uuid::Uuid]) -> GraffitiResult<()>;
}

/// Trait for proof construction through environmental reality
#[async_trait]
pub trait ProofConstruction: Send + Sync {
    /// Construct proof from environmental state
    async fn construct_proof(
        &self,
        query: &Query,
        environmental_state: &EnvironmentalState,
    ) -> GraffitiResult<Proof>;

    /// Validate proof through perturbation testing
    async fn validate_proof(&self, proof: &Proof) -> GraffitiResult<f64>;

    /// Generate alternative proofs
    async fn generate_alternatives(&self, query: &Query) -> GraffitiResult<Vec<Proof>>;

    /// Optimize proof through thermodynamic principles
    async fn optimize_thermodynamically(&self, proof: &Proof) -> GraffitiResult<Proof>;
}

/// Trait for perturbation validation of responses
#[async_trait]
pub trait PerturbationValidation: Send + Sync {
    /// Run systematic perturbation tests
    async fn run_perturbation_tests(&self, proof: &Proof) -> GraffitiResult<f64>;

    /// Test word removal stability
    async fn test_word_removal(&self, content: &str) -> GraffitiResult<f64>;

    /// Test positional rearrangement stability
    async fn test_rearrangement(&self, content: &str) -> GraffitiResult<f64>;

    /// Test semantic substitution stability
    async fn test_semantic_substitution(&self, content: &str) -> GraffitiResult<f64>;

    /// Calculate overall stability score
    async fn calculate_stability(&self, test_results: &[f64]) -> GraffitiResult<f64>;
}

/// Trait for resolution platform creation and management
#[async_trait]
pub trait ResolutionPlatform: Send + Sync {
    /// Create resolution platform for point
    async fn create_platform(&self, point: Point) -> GraffitiResult<Resolution>;

    /// Add evidence to platform
    async fn add_evidence(
        &mut self,
        resolution_id: uuid::Uuid,
        evidence: crate::Evidence,
        is_affirmation: bool,
    ) -> GraffitiResult<()>;

    /// Calculate emerging consensus
    async fn calculate_consensus(&self, resolution_id: uuid::Uuid) -> GraffitiResult<crate::ProbabilisticConsensus>;

    /// Update platform with new evidence
    async fn update_platform(&mut self, resolution_id: uuid::Uuid) -> GraffitiResult<()>;
}

/// Main search engine trait that orchestrates all components
#[async_trait]
pub trait SearchEngine: Send + Sync {
    /// Process query and return complete response
    async fn search(&self, query: Query) -> GraffitiResult<SearchResponse>;

    /// Get engine capabilities
    fn capabilities(&self) -> Vec<String>;

    /// Check engine health
    async fn health_check(&self) -> GraffitiResult<HealthStatus>;

    /// Get performance metrics
    async fn metrics(&self) -> GraffitiResult<PerformanceMetrics>;
}

/// Engine health status
#[derive(Debug, Clone, PartialEq)]
pub struct HealthStatus {
    pub overall_status: ComponentStatus,
    pub environmental_measurement: ComponentStatus,
    pub atmospheric_processing: ComponentStatus,
    pub temporal_coordination: ComponentStatus,
    pub s_entropy_navigation: ComponentStatus,
    pub bmd_processing: ComponentStatus,
    pub proof_construction: ComponentStatus,
    pub perturbation_validation: ComponentStatus,
}

/// Component status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
    Unknown,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceMetrics {
    pub queries_processed: u64,
    pub average_response_time: Duration,
    pub atmospheric_processors_active: u64,
    pub environmental_dimensions_measured: u32,
    pub s_entropy_navigations_successful: u64,
    pub temporal_fragments_coordinated: u64,
    pub proofs_constructed: u64,
    pub perturbation_tests_passed: u64,
    pub system_load: f64,
    pub memory_usage: f64,
    pub error_rate: f64,
}

/// Trait for system configuration management
pub trait Configuration: Send + Sync {
    /// Get configuration value
    fn get<T>(&self, key: &str) -> GraffitiResult<T>
    where
        T: std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Display;

    /// Set configuration value
    fn set(&mut self, key: &str, value: &str) -> GraffitiResult<()>;

    /// Get all configuration keys
    fn keys(&self) -> Vec<String>;

    /// Validate configuration
    fn validate(&self) -> GraffitiResult<()>;
}

/// Trait for logging and observability
#[async_trait]
pub trait Observatory: Send + Sync {
    /// Log query processing event
    async fn log_query(&self, query_id: QueryId, event: QueryEvent) -> GraffitiResult<()>;

    /// Log environmental measurement
    async fn log_environmental(&self, state: &EnvironmentalState) -> GraffitiResult<()>;

    /// Log atmospheric processing event
    async fn log_atmospheric(&self, event: AtmosphericEvent) -> GraffitiResult<()>;

    /// Log performance metrics
    async fn log_metrics(&self, metrics: &PerformanceMetrics) -> GraffitiResult<()>;

    /// Export metrics for monitoring systems
    async fn export_metrics(&self) -> GraffitiResult<HashMap<String, f64>>;
}

/// Query processing events for observability
#[derive(Debug, Clone)]
pub enum QueryEvent {
    Received,
    EnvironmentalMeasurement,
    AtmosphericProcessing,
    TemporalCoordination,
    SEntropyNavigation,
    BMDProcessing,
    ProofConstruction,
    PerturbationValidation,
    ResponseGenerated,
    Error { error: String },
}

/// Atmospheric processing events
#[derive(Debug, Clone)]
pub enum AtmosphericEvent {
    NetworkInitialized,
    MolecularConsensusStarted,
    MolecularConsensusCompleted { consensus_score: f64 },
    ProcessorFailure { processor_id: u64 },
    NetworkOptimization,
}

/// Trait for caching to improve performance
#[async_trait]
pub trait Cache: Send + Sync {
    /// Get cached value
    async fn get<T>(&self, key: &str) -> GraffitiResult<Option<T>>
    where
        T: serde::de::DeserializeOwned;

    /// Set cached value with expiration
    async fn set<T>(&self, key: &str, value: &T, ttl: Duration) -> GraffitiResult<()>
    where
        T: serde::Serialize;

    /// Remove cached value
    async fn remove(&self, key: &str) -> GraffitiResult<()>;

    /// Clear all cached values
    async fn clear(&self) -> GraffitiResult<()>;

    /// Get cache statistics
    async fn stats(&self) -> GraffitiResult<CacheStats>;
}

/// Cache statistics
#[derive(Debug, Clone, PartialEq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: u64,
    pub memory_usage: u64,
}

/// Trait for persistent storage
#[async_trait]
pub trait Storage: Send + Sync {
    /// Store proof permanently
    async fn store_proof(&self, proof: &Proof) -> GraffitiResult<()>;

    /// Retrieve proof by ID
    async fn get_proof(&self, id: uuid::Uuid) -> GraffitiResult<Option<Proof>>;

    /// Store resolution platform
    async fn store_resolution(&self, resolution: &Resolution) -> GraffitiResult<()>;

    /// Query proofs by content similarity
    async fn query_similar_proofs(&self, query: &str, limit: usize) -> GraffitiResult<Vec<Proof>>;

    /// Store environmental measurement for analysis
    async fn store_environmental(&self, state: &EnvironmentalState) -> GraffitiResult<()>;

    /// Get storage statistics
    async fn storage_stats(&self) -> GraffitiResult<StorageStats>;
}

/// Storage statistics
#[derive(Debug, Clone, PartialEq)]
pub struct StorageStats {
    pub total_proofs: u64,
    pub total_resolutions: u64,
    pub total_environmental_measurements: u64,
    pub storage_size: u64,
    pub last_cleanup: std::time::SystemTime,
}
