//! Environmental state integration and validation engine

use graffiti_core::*;
use std::collections::HashMap;
use std::time::SystemTime;

/// Integration engine that combines twelve-dimensional measurements into coherent environmental state
pub struct IntegrationEngine {
    coherence_threshold: f64,
    dimensional_weights: HashMap<String, f64>,
    state_validator: StateValidator,
    uniqueness_detector: UniquenessDetector,
}

impl IntegrationEngine {
    pub fn new() -> Self {
        let mut dimensional_weights = HashMap::new();
        
        // Set weights for dimensional integration based on their contribution to environmental uniqueness
        dimensional_weights.insert("biometric".to_string(), 0.15);
        dimensional_weights.insert("spatial".to_string(), 0.12);
        dimensional_weights.insert("atmospheric".to_string(), 0.18); // High weight - critical for molecular processing
        dimensional_weights.insert("cosmic".to_string(), 0.08);
        dimensional_weights.insert("temporal".to_string(), 0.14); // High weight - critical for coordination
        dimensional_weights.insert("hydrodynamic".to_string(), 0.06);
        dimensional_weights.insert("geological".to_string(), 0.05);
        dimensional_weights.insert("quantum".to_string(), 0.10); // Moderate weight - quantum coherence important
        dimensional_weights.insert("computational".to_string(), 0.04);
        dimensional_weights.insert("acoustic".to_string(), 0.03);
        dimensional_weights.insert("ultrasonic".to_string(), 0.02);
        dimensional_weights.insert("visual".to_string(), 0.03);
        
        Self {
            coherence_threshold: 0.85,
            dimensional_weights,
            state_validator: StateValidator::new(),
            uniqueness_detector: UniquenessDetector::new(),
        }
    }

    pub async fn validate_state(&self, state: &EnvironmentalState) -> GraffitiResult<()> {
        tracing::debug!("Validating environmental state across twelve dimensions");
        
        // 1. Check dimensional coherence
        self.validate_dimensional_coherence(state).await?;
        
        // 2. Validate state consistency
        self.state_validator.validate_consistency(state).await?;
        
        // 3. Assess environmental uniqueness
        let uniqueness = self.uniqueness_detector.calculate_uniqueness(state).await?;
        if uniqueness < environmental::UNIQUENESS_THRESHOLD {
            return Err(GraffitiError::EnvironmentalMeasurement {
                message: format!("Environmental state not unique enough: {:.3} < {:.3}", 
                    uniqueness, environmental::UNIQUENESS_THRESHOLD),
            });
        }
        
        // 4. Check temporal validity
        let age = SystemTime::now()
            .duration_since(state.timestamp)
            .map_err(|e| GraffitiError::EnvironmentalMeasurement {
                message: format!("Invalid timestamp: {}", e),
            })?;
        
        if age > environmental::MAX_STATE_AGE {
            return Err(GraffitiError::EnvironmentalMeasurement {
                message: format!("Environmental state too old: {:?} > {:?}", age, environmental::MAX_STATE_AGE),
            });
        }
        
        tracing::debug!("Environmental state validation successful (uniqueness: {:.6})", uniqueness);
        Ok(())
    }

    async fn validate_dimensional_coherence(&self, state: &EnvironmentalState) -> GraffitiResult<()> {
        // Check cross-dimensional relationships for physical consistency
        
        // 1. Atmospheric-Spatial coherence
        if state.atmospheric.pressure < 50000.0 && state.spatial.elevation < 5000.0 {
            return Err(GraffitiError::EnvironmentalMeasurement {
                message: "Atmospheric pressure too low for elevation".to_string(),
            });
        }
        
        // 2. Temperature-Humidity coherence
        if state.atmospheric.temperature < 273.15 && state.atmospheric.humidity > 80.0 {
            tracing::warn!("Unusual humidity for sub-freezing temperature");
        }
        
        // 3. Quantum-Atmospheric coherence
        if state.quantum.quantum_coherence > 0.9 && state.atmospheric.molecular_density.n2_density < 0.1 {
            tracing::warn!("High quantum coherence in low-density atmosphere");
        }
        
        // 4. Temporal-Biometric coherence
        let circadian_alignment = self.check_circadian_alignment(
            state.temporal.circadian_phase,
            state.biometric.physiological_arousal
        );
        if !circadian_alignment {
            tracing::info!("Biometric state not aligned with circadian rhythm");
        }
        
        Ok(())
    }

    fn check_circadian_alignment(&self, circadian_phase: f64, physiological_arousal: f64) -> bool {
        // Check if physiological arousal aligns with expected circadian rhythm
        let expected_arousal = if circadian_phase > 0.25 && circadian_phase < 0.75 {
            // Daytime - expect higher arousal
            0.6
        } else {
            // Nighttime - expect lower arousal
            0.3
        };
        
        (physiological_arousal - expected_arousal).abs() < 0.4
    }

    pub async fn calculate_environmental_signature(&self, state: &EnvironmentalState) -> GraffitiResult<String> {
        // Create unique environmental signature for proof construction
        let mut signature_components = Vec::new();
        
        // Add weighted dimensional components
        signature_components.push(format!("B{:.3}", state.biometric.cognitive_load));
        signature_components.push(format!("S{:.3}", state.spatial.gravitational_field / physics::EARTH_GRAVITY));
        signature_components.push(format!("A{:.3}", state.atmospheric.pressure / physics::ATMOSPHERIC_PRESSURE_SEA_LEVEL));
        signature_components.push(format!("C{:.3}", state.cosmic.solar_activity));
        signature_components.push(format!("T{:.6}", state.temporal.precision_by_difference));
        signature_components.push(format!("H{:.3}", state.hydrodynamic.local_humidity / 100.0));
        signature_components.push(format!("G{:.3}", state.geological.seismic_activity / 10.0));
        signature_components.push(format!("Q{:.6}", state.quantum.quantum_coherence));
        signature_components.push(format!("P{:.3}", state.computational.processing_load));
        signature_components.push(format!("AC{:.3}", state.acoustic.ambient_noise_level / 100.0));
        signature_components.push(format!("U{:.3}", state.ultrasonic.ultrasonic_reflectivity));
        signature_components.push(format!("V{:.3}", state.visual.illuminance / 1000.0));
        
        Ok(signature_components.join(":"))
    }
}

/// Validates environmental state consistency and physical plausibility
pub struct StateValidator {
    physical_bounds: HashMap<String, (f64, f64)>,
}

impl StateValidator {
    fn new() -> Self {
        let mut physical_bounds = HashMap::new();
        
        // Set physical bounds for measurements
        physical_bounds.insert("atmospheric_pressure".to_string(), (10000.0, 120000.0)); // Pa
        physical_bounds.insert("temperature".to_string(), (173.15, 373.15)); // K (100°C to 100°C)
        physical_bounds.insert("humidity".to_string(), (0.0, 100.0)); // %
        physical_bounds.insert("gravitational_field".to_string(), (9.0, 10.5)); // m/s²
        physical_bounds.insert("quantum_coherence".to_string(), (0.0, 1.0));
        physical_bounds.insert("processing_load".to_string(), (0.0, 1.0));
        
        Self {
            physical_bounds,
        }
    }

    async fn validate_consistency(&self, state: &EnvironmentalState) -> GraffitiResult<()> {
        // Validate atmospheric measurements
        self.check_bounds("atmospheric_pressure", state.atmospheric.pressure)?;
        self.check_bounds("temperature", state.atmospheric.temperature)?;
        self.check_bounds("humidity", state.atmospheric.humidity)?;
        
        // Validate spatial measurements
        self.check_bounds("gravitational_field", state.spatial.gravitational_field)?;
        
        // Validate quantum measurements
        self.check_bounds("quantum_coherence", state.quantum.quantum_coherence)?;
        
        // Validate computational measurements
        self.check_bounds("processing_load", state.computational.processing_load)?;
        
        // Check molecular density consistency
        let total_density = state.atmospheric.molecular_density.n2_density
            + state.atmospheric.molecular_density.o2_density
            + state.atmospheric.molecular_density.h2o_density;
        
        if total_density <= 0.0 {
            return Err(GraffitiError::EnvironmentalMeasurement {
                message: "Total molecular density must be positive".to_string(),
            });
        }
        
        Ok(())
    }

    fn check_bounds(&self, parameter: &str, value: f64) -> GraffitiResult<()> {
        if let Some((min, max)) = self.physical_bounds.get(parameter) {
            if value < *min || value > *max {
                return Err(GraffitiError::EnvironmentalMeasurement {
                    message: format!("{} out of bounds: {:.3} not in [{:.3}, {:.3}]", 
                        parameter, value, min, max),
                });
            }
        }
        Ok(())
    }
}

/// Detects environmental uniqueness for proof construction
pub struct UniquenessDetector {
    entropy_calculator: EntropyCalculator,
    uniqueness_history: Vec<f64>,
    max_history_size: usize,
}

impl UniquenessDetector {
    fn new() -> Self {
        Self {
            entropy_calculator: EntropyCalculator::new(),
            uniqueness_history: Vec::new(),
            max_history_size: 1000,
        }
    }

    async fn calculate_uniqueness(&mut self, state: &EnvironmentalState) -> GraffitiResult<f64> {
        // Calculate multi-dimensional environmental entropy
        let dimensional_entropies = self.entropy_calculator.calculate_dimensional_entropies(state).await?;
        
        // Combine entropies with weights
        let mut total_entropy = 0.0;
        let mut total_weight = 0.0;
        
        let dimension_names = environmental::DIMENSION_NAMES;
        for (i, &dimension) in dimension_names.iter().enumerate() {
            if i < dimensional_entropies.len() {
                let weight = self.get_dimension_weight(dimension);
                total_entropy += dimensional_entropies[i] * weight;
                total_weight += weight;
            }
        }
        
        let average_entropy = if total_weight > 0.0 {
            total_entropy / total_weight
        } else {
            0.0
        };
        
        // Convert entropy to uniqueness (higher entropy = more unique)
        let uniqueness = average_entropy.tanh(); // Squash to [0,1] range
        
        // Update history
        self.uniqueness_history.push(uniqueness);
        if self.uniqueness_history.len() > self.max_history_size {
            self.uniqueness_history.remove(0);
        }
        
        // Enhance uniqueness based on temporal variation
        let temporal_uniqueness = self.calculate_temporal_uniqueness()?;
        let enhanced_uniqueness = (uniqueness + temporal_uniqueness * 0.2).min(1.0);
        
        Ok(enhanced_uniqueness)
    }

    fn get_dimension_weight(&self, dimension: &str) -> f64 {
        match dimension {
            "atmospheric" => 0.2,
            "temporal" => 0.18,
            "quantum" => 0.15,
            "biometric" => 0.12,
            "spatial" => 0.1,
            _ => 0.05,
        }
    }

    fn calculate_temporal_uniqueness(&self) -> GraffitiResult<f64> {
        if self.uniqueness_history.len() < 2 {
            return Ok(0.5); // Default uniqueness
        }

        // Calculate variation in recent uniqueness values
        let recent_values = if self.uniqueness_history.len() > 10 {
            &self.uniqueness_history[self.uniqueness_history.len() - 10..]
        } else {
            &self.uniqueness_history
        };

        let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let variance = recent_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent_values.len() as f64;
        
        // Higher variance means more temporal uniqueness
        Ok(variance.sqrt().min(1.0))
    }
}

/// Calculates entropy for environmental dimensions
pub struct EntropyCalculator;

impl EntropyCalculator {
    fn new() -> Self {
        Self
    }

    async fn calculate_dimensional_entropies(&self, state: &EnvironmentalState) -> GraffitiResult<Vec<f64>> {
        let mut entropies = Vec::new();
        
        // Calculate entropy for each dimension based on its internal structure
        entropies.push(self.calculate_biometric_entropy(&state.biometric));
        entropies.push(self.calculate_spatial_entropy(&state.spatial));
        entropies.push(self.calculate_atmospheric_entropy(&state.atmospheric));
        entropies.push(self.calculate_cosmic_entropy(&state.cosmic));
        entropies.push(self.calculate_temporal_entropy(&state.temporal));
        entropies.push(self.calculate_hydrodynamic_entropy(&state.hydrodynamic));
        entropies.push(self.calculate_geological_entropy(&state.geological));
        entropies.push(self.calculate_quantum_entropy(&state.quantum));
        entropies.push(self.calculate_computational_entropy(&state.computational));
        entropies.push(self.calculate_acoustic_entropy(&state.acoustic));
        entropies.push(self.calculate_ultrasonic_entropy(&state.ultrasonic));
        entropies.push(self.calculate_visual_entropy(&state.visual));
        
        Ok(entropies)
    }

    fn calculate_biometric_entropy(&self, bio: &BiometricDimension) -> f64 {
        let values = [bio.physiological_arousal, bio.cognitive_load, bio.attention_state, bio.emotional_valence.abs()];
        self.shannon_entropy(&values)
    }

    fn calculate_spatial_entropy(&self, spatial: &SpatialDimension) -> f64 {
        let values = [
            spatial.position.x.abs() / 1000.0,
            spatial.position.y.abs() / 1000.0,
            spatial.position.z / 10.0,
            spatial.gravitational_field / physics::EARTH_GRAVITY,
            spatial.magnetic_field.magnitude() / 50.0,
            spatial.elevation / 9000.0,
        ];
        self.shannon_entropy(&values)
    }

    fn calculate_atmospheric_entropy(&self, atm: &AtmosphericDimension) -> f64 {
        let values = [
            atm.pressure / physics::ATMOSPHERIC_PRESSURE_SEA_LEVEL,
            atm.humidity / 100.0,
            (atm.temperature - 273.15) / 100.0 + 0.5, // Normalize around 0°C
            atm.molecular_density.n2_density * 1e20,
            atm.molecular_density.o2_density * 1e20,
            atm.air_quality_index / 500.0,
        ];
        self.shannon_entropy(&values)
    }

    fn calculate_cosmic_entropy(&self, cosmic: &CosmicDimension) -> f64 {
        let values = [
            cosmic.solar_activity,
            cosmic.cosmic_radiation / 10.0,
            cosmic.geomagnetic_activity,
            cosmic.solar_wind.magnitude() / 1000.0,
        ];
        self.shannon_entropy(&values)
    }

    fn calculate_temporal_entropy(&self, temporal: &TemporalDimension) -> f64 {
        let values = [
            temporal.circadian_phase,
            temporal.seasonal_phase,
            temporal.lunar_phase,
            (temporal.precision_by_difference * 1e30).fract(), // Use fractional part of high-precision value
        ];
        self.shannon_entropy(&values)
    }

    fn calculate_hydrodynamic_entropy(&self, hydro: &HydrodynamicDimension) -> f64 {
        let values = [
            hydro.local_humidity / 100.0,
            hydro.water_vapor_pressure / 5000.0,
            hydro.fluid_dynamics.magnitude(),
            hydro.hydrostatic_pressure / 103000.0,
        ];
        self.shannon_entropy(&values)
    }

    fn calculate_geological_entropy(&self, geo: &GeologicalDimension) -> f64 {
        let mut values = vec![
            geo.seismic_activity / 10.0,
            geo.tectonic_stress / 100.0,
            geo.earth_currents.magnitude(),
        ];
        
        // Add mineral composition entropy
        for value in geo.mineral_composition.values() {
            values.push(*value);
        }
        
        self.shannon_entropy(&values)
    }

    fn calculate_quantum_entropy(&self, quantum: &QuantumDimension) -> f64 {
        let values = [
            quantum.quantum_coherence,
            quantum.entanglement_density,
            quantum.vacuum_fluctuations,
            quantum.quantum_noise * 10.0, // Scale up noise
        ];
        self.shannon_entropy(&values)
    }

    fn calculate_computational_entropy(&self, comp: &ComputationalDimension) -> f64 {
        let values = [
            comp.processing_load,
            comp.memory_usage,
            comp.network_latency * 100.0,
            comp.system_entropy,
        ];
        self.shannon_entropy(&values)
    }

    fn calculate_acoustic_entropy(&self, acoustic: &AcousticDimension) -> f64 {
        let mut values = vec![
            acoustic.ambient_noise_level / 100.0,
            acoustic.acoustic_impedance / 500.0,
            acoustic.sound_velocity / 400.0,
        ];
        
        // Add frequency spectrum entropy
        for &amplitude in acoustic.frequency_spectrum.iter().take(8) {
            values.push(amplitude);
        }
        
        self.shannon_entropy(&values)
    }

    fn calculate_ultrasonic_entropy(&self, ultra: &UltrasonicDimension) -> f64 {
        let mut values = vec![
            ultra.ultrasonic_reflectivity,
            ultra.material_density / 3000.0,
        ];
        
        // Add geometric features and distance measurements
        values.extend(ultra.geometric_features.iter().take(4));
        values.extend(ultra.distance_measurements.iter().take(4).map(|d| d / 10.0));
        
        self.shannon_entropy(&values)
    }

    fn calculate_visual_entropy(&self, visual: &VisualDimension) -> f64 {
        let mut values = vec![
            (visual.illuminance / 100000.0).min(1.0),
            visual.color_temperature / 8000.0,
            visual.visual_complexity,
        ];
        
        // Add spectral composition entropy
        for &intensity in visual.spectral_composition.iter().take(8) {
            values.push(intensity);
        }
        
        self.shannon_entropy(&values)
    }

    fn shannon_entropy(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        // Normalize values to create probability distribution
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
        
        entropy
    }
}
