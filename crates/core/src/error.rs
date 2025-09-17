//! Error types for the Graffiti Search engine

use thiserror::Error;
use std::time::SystemTime;
use crate::QueryId;

/// Main error type for the Graffiti Search engine
#[derive(Error, Debug)]
pub enum GraffitiError {
    #[error("Environmental measurement error: {message}")]
    EnvironmentalMeasurement { message: String },

    #[error("Atmospheric processing error: {message}")]
    AtmosphericProcessing { message: String },

    #[error("Temporal coordination error: {message}")]
    TemporalCoordination { message: String },

    #[error("S-entropy navigation error: {message}")]
    SEntropyNavigation { message: String },

    #[error("BMD processing error: {message}")]
    BMDProcessing { message: String },

    #[error("Proof construction error: {message}")]
    ProofConstruction { message: String },

    #[error("Perturbation validation error: {message}")]
    PerturbationValidation { message: String },

    #[error("Query processing error for {query_id:?}: {message}")]
    QueryProcessing { query_id: QueryId, message: String },

    #[error("System impossibility detected: {impossibility_type}")]
    SystematicImpossibility { impossibility_type: ImpossibilityType },

    #[error("Configuration error: {field} - {message}")]
    Configuration { field: String, message: String },

    #[error("Network error: {message}")]
    Network { message: String },

    #[error("Serialization error: {message}")]
    Serialization { message: String },

    #[error("Database error: {message}")]
    Database { message: String },

    #[error("Permission denied: {operation}")]
    PermissionDenied { operation: String },

    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("System overload: {component}")]
    SystemOverload { component: String },

    #[error("Quantum measurement error: {message}")]
    QuantumMeasurement { message: String },
}

/// Types of systematic impossibilities the system may encounter
#[derive(Debug, Clone, PartialEq)]
pub enum ImpossibilityType {
    MeaningCreation,
    MetaKnowledgeInfiniteRegress,
    TemporalPredeterminationAccess,
    AbsoluteCoordinatePrecision,
    OscillatoryConvergenceControl,
    QuantumCoherenceMaintenance,
    ConsciousnessSubstrateIndependence,
    CollectiveTruthVerification,
    ThermodynamicReversibility,
    ZeroTemporalDelayUnderstanding,
    InformationConservation,
    TemporalDimensionFundamentality,
}

/// Result type alias for Graffiti operations
pub type GraffitiResult<T> = Result<T, GraffitiError>;

/// Error context for debugging and monitoring
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub timestamp: SystemTime,
    pub component: String,
    pub operation: String,
    pub environmental_state_hash: Option<String>,
    pub query_id: Option<QueryId>,
    pub additional_metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            timestamp: SystemTime::now(),
            component: component.to_string(),
            operation: operation.to_string(),
            environmental_state_hash: None,
            query_id: None,
            additional_metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_query_id(mut self, query_id: QueryId) -> Self {
        self.query_id = Some(query_id);
        self
    }

    pub fn with_environmental_hash(mut self, hash: String) -> Self {
        self.environmental_state_hash = Some(hash);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.additional_metadata.insert(key, value);
        self
    }
}

/// Specialized error for environmental measurement failures
#[derive(Error, Debug)]
pub enum EnvironmentalError {
    #[error("Sensor {sensor_id} failed: {reason}")]
    SensorFailure { sensor_id: String, reason: String },

    #[error("Dimension {dimension} measurement out of bounds: {value}")]
    OutOfBounds { dimension: String, value: f64 },

    #[error("Calibration error for {dimension}: {message}")]
    CalibrationError { dimension: String, message: String },

    #[error("Environmental state inconsistent: {details}")]
    StateInconsistency { details: String },

    #[error("Twelve-dimensional measurement incomplete: missing {missing_dimensions:?}")]
    IncompleteMeasurement { missing_dimensions: Vec<String> },
}

/// Specialized error for atmospheric processing
#[derive(Error, Debug)]
pub enum AtmosphericError {
    #[error("Molecular density calculation failed: {reason}")]
    MolecularDensityError { reason: String },

    #[error("Atmospheric processor {processor_id} unresponsive")]
    ProcessorUnresponsive { processor_id: u64 },

    #[error("Network consensus failed: {participating_molecules} molecules participated")]
    ConsensusFailure { participating_molecules: u64 },

    #[error("Molecular oscillation out of sync: frequency {frequency} Hz")]
    OscillationError { frequency: f64 },
}

/// Specialized error for proof validation
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Perturbation test failed: stability {stability}, threshold {threshold}")]
    PerturbationFailure { stability: f64, threshold: f64 },

    #[error("Proof inconsistency detected at step {step}: {reason}")]
    ProofInconsistency { step: usize, reason: String },

    #[error("Insufficient evidence: confidence {confidence}, minimum required {minimum}")]
    InsufficientEvidence { confidence: f64, minimum: f64 },

    #[error("Logical gap in proof: {description}")]
    LogicalGap { description: String },
}

// Conversion implementations for nested errors
impl From<EnvironmentalError> for GraffitiError {
    fn from(err: EnvironmentalError) -> Self {
        GraffitiError::EnvironmentalMeasurement {
            message: err.to_string(),
        }
    }
}

impl From<AtmosphericError> for GraffitiError {
    fn from(err: AtmosphericError) -> Self {
        GraffitiError::AtmosphericProcessing {
            message: err.to_string(),
        }
    }
}

impl From<ValidationError> for GraffitiError {
    fn from(err: ValidationError) -> Self {
        GraffitiError::PerturbationValidation {
            message: err.to_string(),
        }
    }
}
