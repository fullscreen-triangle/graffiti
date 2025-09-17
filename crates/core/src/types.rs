//! Core types for the Graffiti Search engine
//!
//! This module defines the fundamental types used throughout the revolutionary
//! proof-based search engine architecture.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;
use nalgebra::Vector3;

/// Query identifier for tracking queries through the system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(pub Uuid);

impl QueryId {
    /// Generate a new unique query ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Environmental state vector representing twelve-dimensional measurement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentalState {
    pub timestamp: SystemTime,
    pub biometric: BiometricDimension,
    pub spatial: SpatialDimension,
    pub atmospheric: AtmosphericDimension,
    pub cosmic: CosmicDimension,
    pub temporal: TemporalDimension,
    pub hydrodynamic: HydrodynamicDimension,
    pub geological: GeologicalDimension,
    pub quantum: QuantumDimension,
    pub computational: ComputationalDimension,
    pub acoustic: AcousticDimension,
    pub ultrasonic: UltrasonicDimension,
    pub visual: VisualDimension,
}

/// Biometric environmental state detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BiometricDimension {
    pub physiological_arousal: f64,
    pub cognitive_load: f64,
    pub attention_state: f64,
    pub emotional_valence: f64,
}

/// Spatial positioning and gravitational field awareness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialDimension {
    pub position: Vector3<f64>,
    pub gravitational_field: f64,
    pub magnetic_field: Vector3<f64>,
    pub elevation: f64,
}

/// Atmospheric molecular configuration sensing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtmosphericDimension {
    pub pressure: f64,
    pub humidity: f64,
    pub temperature: f64,
    pub molecular_density: MolecularDensity,
    pub air_quality_index: f64,
}

/// Molecular density distribution for atmospheric processing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MolecularDensity {
    pub n2_density: f64,
    pub o2_density: f64,
    pub h2o_density: f64,
    pub trace_gases: HashMap<String, f64>,
}

/// Cosmic environmental condition awareness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmicDimension {
    pub solar_activity: f64,
    pub cosmic_radiation: f64,
    pub geomagnetic_activity: f64,
    pub solar_wind: Vector3<f64>,
}

/// Temporal rhythms and chronobiology
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalDimension {
    pub circadian_phase: f64,
    pub seasonal_phase: f64,
    pub lunar_phase: f64,
    pub precision_by_difference: f64,
}

/// Hydrodynamic state sensing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HydrodynamicDimension {
    pub local_humidity: f64,
    pub water_vapor_pressure: f64,
    pub fluid_dynamics: Vector3<f64>,
    pub hydrostatic_pressure: f64,
}

/// Geological crustal condition awareness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeologicalDimension {
    pub seismic_activity: f64,
    pub mineral_composition: HashMap<String, f64>,
    pub tectonic_stress: f64,
    pub earth_currents: Vector3<f64>,
}

/// Quantum environmental state detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumDimension {
    pub quantum_coherence: f64,
    pub entanglement_density: f64,
    pub vacuum_fluctuations: f64,
    pub quantum_noise: f64,
}

/// Computational system configuration awareness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputationalDimension {
    pub processing_load: f64,
    pub memory_usage: f64,
    pub network_latency: f64,
    pub system_entropy: f64,
}

/// Acoustic environmental mapping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticDimension {
    pub ambient_noise_level: f64,
    pub frequency_spectrum: Vec<f64>,
    pub acoustic_impedance: f64,
    pub sound_velocity: f64,
}

/// Ultrasonic environmental analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UltrasonicDimension {
    pub ultrasonic_reflectivity: f64,
    pub material_density: f64,
    pub geometric_features: Vec<f64>,
    pub distance_measurements: Vec<f64>,
}

/// Visual photonic environmental state detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisualDimension {
    pub illuminance: f64,
    pub color_temperature: f64,
    pub spectral_composition: Vec<f64>,
    pub visual_complexity: f64,
}

/// Gas molecular information element
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationMolecule {
    pub energy: f64,
    pub entropy: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub velocity: Vector3<f64>,
    pub content: String,
    pub significance: f64,
}

/// Point representing irreducible semantic content
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub id: Uuid,
    pub content: String,
    pub confidence: f64,
    pub interpretations: Vec<Interpretation>,
    pub context_dependencies: HashMap<String, f64>,
    pub semantic_bounds: (f64, f64),
    pub created_at: SystemTime,
}

/// Interpretation of a point with probability
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interpretation {
    pub meaning: String,
    pub probability: f64,
    pub evidence: Vec<String>,
    pub context: String,
}

/// Resolution as debate platform for points
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resolution {
    pub id: Uuid,
    pub point: Point,
    pub affirmations: Vec<Evidence>,
    pub contentions: Vec<Evidence>,
    pub consensus: ProbabilisticConsensus,
    pub debate_status: DebateStatus,
    pub created_at: SystemTime,
    pub last_updated: SystemTime,
}

/// Evidence supporting or challenging a point
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub content: String,
    pub source: String,
    pub reliability: f64,
    pub relevance: f64,
    pub evidence_type: EvidenceType,
    pub timestamp: SystemTime,
}

/// Types of evidence in the debate platform
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceType {
    Mathematical,
    Empirical,
    Logical,
    Experimental,
    Theoretical,
    Historical,
    Observational,
}

/// Probabilistic consensus emerging from debate
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbabilisticConsensus {
    pub confidence_distribution: HashMap<String, f64>,
    pub overall_confidence: f64,
    pub uncertainty_bounds: (f64, f64),
    pub minority_positions: Vec<MinorityPosition>,
}

/// Minority position preserved in consensus
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinorityPosition {
    pub position: String,
    pub support_level: f64,
    pub reasoning: String,
    pub evidence: Vec<Evidence>,
}

/// Status of debate platform
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DebateStatus {
    Open,
    Active,
    Converging,
    Consensus,
    Contested,
}

/// S-entropy coordinates for strategic impossibility navigation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SEntropyCoordinates {
    pub s_knowledge: f64,
    pub s_time: f64,
    pub s_entropy: f64,
    pub strategic_weight: f64,
}

/// Temporal fragment for coordinated information delivery
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalFragment {
    pub id: Uuid,
    pub content: String,
    pub delivery_time: SystemTime,
    pub coherence_window: Duration,
    pub fragment_index: usize,
    pub total_fragments: usize,
}

/// BMD (Biological Maxwell Demon) frame for consciousness processing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BMDFrame {
    pub id: Uuid,
    pub knowledge_component: HashMap<String, f64>,
    pub truth_component: HashMap<String, f64>,
    pub frame_weight: f64,
    pub selection_probability: f64,
    pub last_accessed: SystemTime,
}

/// Complete proof constructed through environmental measurement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proof {
    pub id: Uuid,
    pub query: String,
    pub environmental_signature: EnvironmentalState,
    pub construction_method: ConstructionMethod,
    pub proof_steps: Vec<ProofStep>,
    pub confidence: f64,
    pub validation_status: ValidationStatus,
    pub perturbation_stability: f64,
    pub created_at: SystemTime,
}

/// Method used to construct the proof
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstructionMethod {
    EnvironmentalConstruction,
    AtmosphericMolecular,
    SEntropyNavigation,
    TemporalCoordination,
    BMDFrameSelection,
    ThermodynamicOptimization,
    HybridApproach(Vec<ConstructionMethod>),
}

/// Individual step in proof construction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofStep {
    pub step_number: usize,
    pub content: String,
    pub justification: String,
    pub confidence: f64,
    pub environmental_support: f64,
}

/// Validation status of constructed proof
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Pending,
    Validated,
    PartiallyValidated { issues: Vec<String> },
    Invalid { reasons: Vec<String> },
    UnderReview,
}

/// Search query with environmental context
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub id: QueryId,
    pub content: String,
    pub environmental_context: EnvironmentalState,
    pub user_context: UserContext,
    pub expected_response_type: ResponseType,
    pub urgency: Urgency,
    pub created_at: SystemTime,
}

/// User context for personalized proof construction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserContext {
    pub expertise_level: ExpertiseLevel,
    pub preferred_proof_style: ProofStyle,
    pub context_preferences: HashMap<String, f64>,
    pub historical_queries: Vec<QueryId>,
}

/// Level of user expertise for proof presentation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
    Researcher,
}

/// Style of proof presentation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProofStyle {
    Intuitive,
    Rigorous,
    Computational,
    Visual,
    Narrative,
    Interactive,
}

/// Expected type of response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseType {
    Proof,
    Explanation,
    Calculation,
    Analysis,
    Synthesis,
    Critique,
}

/// Query urgency level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Urgency {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

/// Complete search response with all components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query_id: QueryId,
    pub primary_proof: Proof,
    pub alternative_proofs: Vec<Proof>,
    pub resolution_platforms: Vec<Resolution>,
    pub environmental_analysis: EnvironmentalAnalysis,
    pub confidence_assessment: ConfidenceAssessment,
    pub response_metadata: ResponseMetadata,
}

/// Analysis of environmental factors in proof construction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentalAnalysis {
    pub dominant_factors: Vec<String>,
    pub environmental_uniqueness: f64,
    pub atmospheric_contribution: f64,
    pub temporal_coordination_quality: f64,
}

/// Assessment of response confidence and reliability
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceAssessment {
    pub overall_confidence: f64,
    pub perturbation_stability: f64,
    pub consensus_strength: f64,
    pub uncertainty_factors: Vec<String>,
    pub reliability_metrics: HashMap<String, f64>,
}

/// Metadata about response generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub construction_time: Duration,
    pub atmospheric_processors_used: u64,
    pub environmental_dimensions_active: Vec<String>,
    pub s_entropy_navigations: u32,
    pub temporal_fragments_coordinated: u32,
    pub generation_timestamp: SystemTime,
}
