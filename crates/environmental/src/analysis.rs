//! Environmental analysis framework for proof construction

use graffiti_core::*;
use crate::sensors::SensorNetwork;
use std::collections::HashMap;
use std::time::SystemTime;

/// Framework for analyzing environmental patterns and their suitability for proof construction
pub struct AnalysisFramework {
    pattern_detector: PatternDetector,
    stability_analyzer: StabilityAnalyzer,
    uniqueness_tracker: UniquenessTracker,
    proof_readiness_assessor: ProofReadinessAssessor,
}

impl AnalysisFramework {
    pub fn new() -> Self {
        Self {
            pattern_detector: PatternDetector::new(),
            stability_analyzer: StabilityAnalyzer::new(),
            uniqueness_tracker: UniquenessTracker::new(),
            proof_readiness_assessor: ProofReadinessAssessor::new(),
        }
    }

    pub async fn assess_stability(&self, sensors: &SensorNetwork) -> GraffitiResult<bool> {
        // Check if environmental measurement system is stable
        let stability_score = self.stability_analyzer.assess_sensor_stability(sensors).await?;
        Ok(stability_score > 0.85) // Require 85% stability
    }

    pub async fn analyze_environmental_suitability(
        &mut self,
        state: &EnvironmentalState,
        query_context: &str,
    ) -> GraffitiResult<EnvironmentalAnalysis> {
        tracing::debug!("Analyzing environmental suitability for proof construction");

        // 1. Detect environmental patterns
        let patterns = self.pattern_detector.detect_patterns(state).await?;
        
        // 2. Assess environmental uniqueness
        let uniqueness = self.uniqueness_tracker.track_uniqueness(state).await?;
        
        // 3. Calculate atmospheric contribution potential
        let atmospheric_contribution = self.calculate_atmospheric_contribution(state).await?;
        
        // 4. Assess temporal coordination quality
        let temporal_quality = self.assess_temporal_coordination_quality(state).await?;
        
        // 5. Identify dominant environmental factors
        let dominant_factors = self.identify_dominant_factors(state, &patterns).await?;
        
        Ok(EnvironmentalAnalysis {
            dominant_factors,
            environmental_uniqueness: uniqueness,
            atmospheric_contribution,
            temporal_coordination_quality: temporal_quality,
        })
    }

    async fn calculate_atmospheric_contribution(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        // Calculate how much the atmospheric conditions contribute to proof construction capability
        let molecular_density_factor = (
            state.atmospheric.molecular_density.n2_density * atmospheric::N2_PERCENTAGE / 100.0 +
            state.atmospheric.molecular_density.o2_density * atmospheric::O2_PERCENTAGE / 100.0
        ) / 2.0;
        
        let pressure_factor = (state.atmospheric.pressure / physics::ATMOSPHERIC_PRESSURE_SEA_LEVEL).min(1.2);
        let temperature_factor = if state.atmospheric.temperature > 200.0 && state.atmospheric.temperature < 320.0 {
            1.0 // Optimal temperature range
        } else {
            0.7 // Suboptimal temperature
        };
        
        let humidity_factor = if state.atmospheric.humidity > 30.0 && state.atmospheric.humidity < 70.0 {
            1.0 // Optimal humidity range
        } else {
            0.8 // Suboptimal humidity
        };

        let contribution = molecular_density_factor * pressure_factor * temperature_factor * humidity_factor;
        Ok(contribution.clamp(0.0, 1.0))
    }

    async fn assess_temporal_coordination_quality(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        // Assess how well the temporal conditions support zero-latency coordination
        let precision_quality = if state.temporal.precision_by_difference > temporal::TARGET_PRECISION * 0.1 {
            1.0 // High precision available
        } else {
            0.6 // Lower precision
        };

        let circadian_alignment = (state.temporal.circadian_phase - 0.5).abs(); // Distance from noon
        let circadian_quality = 1.0 - circadian_alignment; // Closer to noon = better

        let temporal_coherence = self.calculate_temporal_coherence(state).await?;

        let quality = (precision_quality * 0.4 + circadian_quality * 0.3 + temporal_coherence * 0.3).clamp(0.0, 1.0);
        Ok(quality)
    }

    async fn calculate_temporal_coherence(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        // Calculate coherence between different temporal measurements
        let temporal_factors = [
            state.temporal.circadian_phase,
            state.temporal.seasonal_phase,
            state.temporal.lunar_phase,
        ];

        // Calculate how well-aligned the temporal factors are
        let mean = temporal_factors.iter().sum::<f64>() / temporal_factors.len() as f64;
        let variance = temporal_factors.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / temporal_factors.len() as f64;

        // Lower variance = higher coherence
        let coherence = 1.0 / (1.0 + variance * 10.0);
        Ok(coherence.clamp(0.0, 1.0))
    }

    async fn identify_dominant_factors(
        &self,
        state: &EnvironmentalState,
        patterns: &[EnvironmentalPattern],
    ) -> GraffitiResult<Vec<String>> {
        let mut factor_scores = HashMap::new();

        // Score based on measurement magnitudes (normalized)
        factor_scores.insert("atmospheric".to_string(), 
            state.atmospheric.pressure / physics::ATMOSPHERIC_PRESSURE_SEA_LEVEL);
        factor_scores.insert("spatial".to_string(),
            state.spatial.gravitational_field / physics::EARTH_GRAVITY);
        factor_scores.insert("temporal".to_string(),
            state.temporal.precision_by_difference / temporal::TARGET_PRECISION);
        factor_scores.insert("quantum".to_string(),
            state.quantum.quantum_coherence);
        factor_scores.insert("biometric".to_string(),
            (state.biometric.cognitive_load + state.biometric.attention_state) / 2.0);

        // Boost scores based on detected patterns
        for pattern in patterns {
            if let Some(score) = factor_scores.get_mut(&pattern.dimension) {
                *score *= 1.0 + pattern.strength;
            }
        }

        // Sort factors by score and take top ones
        let mut sorted_factors: Vec<(String, f64)> = factor_scores.into_iter().collect();
        sorted_factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top 3 factors
        Ok(sorted_factors.into_iter()
            .take(3)
            .map(|(factor, _)| factor)
            .collect())
    }
}

/// Detects patterns in environmental measurements
pub struct PatternDetector {
    pattern_history: Vec<EnvironmentalPattern>,
    max_history: usize,
}

impl PatternDetector {
    fn new() -> Self {
        Self {
            pattern_history: Vec::new(),
            max_history: 1000,
        }
    }

    async fn detect_patterns(&mut self, state: &EnvironmentalState) -> GraffitiResult<Vec<EnvironmentalPattern>> {
        let mut patterns = Vec::new();

        // Detect atmospheric patterns
        if let Some(pattern) = self.detect_atmospheric_pattern(state).await? {
            patterns.push(pattern);
        }

        // Detect temporal patterns
        if let Some(pattern) = self.detect_temporal_pattern(state).await? {
            patterns.push(pattern);
        }

        // Detect quantum coherence patterns
        if let Some(pattern) = self.detect_quantum_pattern(state).await? {
            patterns.push(pattern);
        }

        // Detect biometric patterns
        if let Some(pattern) = self.detect_biometric_pattern(state).await? {
            patterns.push(pattern);
        }

        // Update pattern history
        for pattern in &patterns {
            self.pattern_history.push(pattern.clone());
        }
        if self.pattern_history.len() > self.max_history {
            self.pattern_history.drain(0..self.pattern_history.len() - self.max_history);
        }

        Ok(patterns)
    }

    async fn detect_atmospheric_pattern(&self, state: &EnvironmentalState) -> GraffitiResult<Option<EnvironmentalPattern>> {
        // Detect significant atmospheric molecular concentrations
        let n2_ratio = state.atmospheric.molecular_density.n2_density / 
            (atmospheric::N2_PERCENTAGE / 100.0 * state.atmospheric.pressure / (physics::GAS_CONSTANT * state.atmospheric.temperature));
        
        if (n2_ratio - 1.0).abs() > 0.1 {
            return Ok(Some(EnvironmentalPattern {
                dimension: "atmospheric".to_string(),
                pattern_type: PatternType::MolecularConcentration,
                strength: (n2_ratio - 1.0).abs(),
                confidence: 0.8,
                timestamp: SystemTime::now(),
            }));
        }

        Ok(None)
    }

    async fn detect_temporal_pattern(&self, state: &EnvironmentalState) -> GraffitiResult<Option<EnvironmentalPattern>> {
        // Detect high temporal precision windows
        if state.temporal.precision_by_difference > temporal::TARGET_PRECISION * 10.0 {
            return Ok(Some(EnvironmentalPattern {
                dimension: "temporal".to_string(),
                pattern_type: PatternType::HighPrecision,
                strength: state.temporal.precision_by_difference / temporal::TARGET_PRECISION,
                confidence: 0.9,
                timestamp: SystemTime::now(),
            }));
        }

        Ok(None)
    }

    async fn detect_quantum_pattern(&self, state: &EnvironmentalState) -> GraffitiResult<Option<EnvironmentalPattern>> {
        // Detect high quantum coherence states
        if state.quantum.quantum_coherence > 0.8 {
            return Ok(Some(EnvironmentalPattern {
                dimension: "quantum".to_string(),
                pattern_type: PatternType::HighCoherence,
                strength: state.quantum.quantum_coherence,
                confidence: 0.7, // Quantum measurements have inherent uncertainty
                timestamp: SystemTime::now(),
            }));
        }

        Ok(None)
    }

    async fn detect_biometric_pattern(&self, state: &EnvironmentalState) -> GraffitiResult<Option<EnvironmentalPattern>> {
        // Detect high cognitive load states
        if state.biometric.cognitive_load > 0.8 && state.biometric.attention_state > 0.7 {
            return Ok(Some(EnvironmentalPattern {
                dimension: "biometric".to_string(),
                pattern_type: PatternType::HighCognition,
                strength: (state.biometric.cognitive_load + state.biometric.attention_state) / 2.0,
                confidence: 0.75,
                timestamp: SystemTime::now(),
            }));
        }

        Ok(None)
    }
}

/// Environmental pattern detected in measurements
#[derive(Debug, Clone)]
pub struct EnvironmentalPattern {
    pub dimension: String,
    pub pattern_type: PatternType,
    pub strength: f64,
    pub confidence: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    MolecularConcentration,
    HighPrecision,
    HighCoherence,
    HighCognition,
    ThermalStability,
    PressureAnomaly,
    GeodeticShift,
}

/// Analyzes stability of environmental measurements
pub struct StabilityAnalyzer {
    measurement_history: Vec<f64>,
    stability_window: usize,
}

impl StabilityAnalyzer {
    fn new() -> Self {
        Self {
            measurement_history: Vec::new(),
            stability_window: 10,
        }
    }

    async fn assess_sensor_stability(&mut self, _sensors: &SensorNetwork) -> GraffitiResult<f64> {
        // In a real implementation, this would analyze actual sensor readings over time
        // For now, simulate stability assessment
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let current_stability = rng.gen_range(0.8..0.98);
        
        self.measurement_history.push(current_stability);
        if self.measurement_history.len() > self.stability_window {
            self.measurement_history.remove(0);
        }

        // Calculate stability from variance in recent measurements
        if self.measurement_history.len() < 3 {
            return Ok(current_stability);
        }

        let mean = self.measurement_history.iter().sum::<f64>() / self.measurement_history.len() as f64;
        let variance = self.measurement_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.measurement_history.len() as f64;

        // Lower variance = higher stability
        let stability = (1.0 - variance.sqrt()).max(0.0);
        Ok(stability)
    }
}

/// Tracks environmental uniqueness over time
pub struct UniquenessTracker {
    uniqueness_history: Vec<f64>,
    max_history: usize,
}

impl UniquenessTracker {
    fn new() -> Self {
        Self {
            uniqueness_history: Vec::new(),
            max_history: 100,
        }
    }

    async fn track_uniqueness(&mut self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        // Calculate current uniqueness based on multi-dimensional entropy
        let uniqueness = self.calculate_current_uniqueness(state).await?;

        // Update history
        self.uniqueness_history.push(uniqueness);
        if self.uniqueness_history.len() > self.max_history {
            self.uniqueness_history.remove(0);
        }

        // Apply temporal enhancement based on variation
        let enhanced_uniqueness = if self.uniqueness_history.len() > 1 {
            let recent_mean = self.uniqueness_history.iter().skip(self.uniqueness_history.len().saturating_sub(5))
                .sum::<f64>() / 5.0_f64.min(self.uniqueness_history.len() as f64);
            let enhancement = (uniqueness - recent_mean).abs() * 0.1;
            (uniqueness + enhancement).min(1.0)
        } else {
            uniqueness
        };

        Ok(enhanced_uniqueness)
    }

    async fn calculate_current_uniqueness(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        // Calculate uniqueness based on environmental state diversity
        let mut uniqueness_components = Vec::new();

        // Atmospheric uniqueness
        uniqueness_components.push(state.atmospheric.molecular_density.n2_density.fract());
        uniqueness_components.push(state.atmospheric.pressure.fract());
        
        // Temporal uniqueness
        uniqueness_components.push(state.temporal.precision_by_difference.fract());
        uniqueness_components.push(state.temporal.circadian_phase);
        
        // Quantum uniqueness
        uniqueness_components.push(state.quantum.quantum_coherence);
        uniqueness_components.push(state.quantum.vacuum_fluctuations);
        
        // Spatial uniqueness
        uniqueness_components.push((state.spatial.position.x / 1000.0).fract());
        uniqueness_components.push((state.spatial.position.y / 1000.0).fract());

        // Calculate Shannon entropy of components
        let entropy = self.calculate_shannon_entropy(&uniqueness_components);
        Ok(entropy.tanh()) // Normalize to [0,1]
    }

    fn calculate_shannon_entropy(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let sum: f64 = values.iter().map(|x| x.abs()).sum();
        if sum == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &value in values {
            let p = value.abs() / sum;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        
        entropy / (values.len() as f64).log2() // Normalize
    }
}

/// Assesses readiness for proof construction
pub struct ProofReadinessAssessor;

impl ProofReadinessAssessor {
    fn new() -> Self {
        Self
    }

    pub async fn assess_proof_readiness(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        // Assess multiple factors that contribute to proof construction readiness
        
        // 1. Atmospheric readiness (molecular density and stability)
        let atmospheric_readiness = self.assess_atmospheric_readiness(state).await?;
        
        // 2. Temporal readiness (precision and coordination potential)
        let temporal_readiness = self.assess_temporal_readiness(state).await?;
        
        // 3. Quantum readiness (coherence and low noise)
        let quantum_readiness = self.assess_quantum_readiness(state).await?;
        
        // 4. Environmental uniqueness readiness
        let uniqueness_readiness = if state.timestamp.elapsed().unwrap_or_default() < environmental::MAX_STATE_AGE {
            1.0 // Fresh measurement
        } else {
            0.3 // Stale measurement
        };

        // Combine readiness factors
        let overall_readiness = atmospheric_readiness * 0.3 
            + temporal_readiness * 0.3 
            + quantum_readiness * 0.2 
            + uniqueness_readiness * 0.2;

        Ok(overall_readiness.clamp(0.0, 1.0))
    }

    async fn assess_atmospheric_readiness(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        let molecular_density_score = (state.atmospheric.molecular_density.n2_density 
            + state.atmospheric.molecular_density.o2_density).min(1.0);
        let pressure_score = (state.atmospheric.pressure / physics::ATMOSPHERIC_PRESSURE_SEA_LEVEL).min(1.2);
        let temperature_score = if state.atmospheric.temperature > 250.0 && state.atmospheric.temperature < 320.0 {
            1.0
        } else {
            0.7
        };

        Ok((molecular_density_score * pressure_score * temperature_score).clamp(0.0, 1.0))
    }

    async fn assess_temporal_readiness(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        let precision_score = (state.temporal.precision_by_difference / temporal::TARGET_PRECISION * 0.01).min(1.0);
        let circadian_score = 1.0 - (state.temporal.circadian_phase - 0.5).abs(); // Peak at noon
        
        Ok((precision_score * 0.7 + circadian_score * 0.3).clamp(0.0, 1.0))
    }

    async fn assess_quantum_readiness(&self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        let coherence_score = state.quantum.quantum_coherence;
        let noise_score = 1.0 - state.quantum.quantum_noise.min(1.0);
        
        Ok((coherence_score * 0.8 + noise_score * 0.2).clamp(0.0, 1.0))
    }
}
