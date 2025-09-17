//! Atmospheric molecular network for distributed processing

use graffiti_core::*;
use std::collections::HashMap;
use std::time::SystemTime;
use nalgebra::Vector3;

/// Network coordinator for Earth's 10^44 atmospheric molecules
pub struct MolecularNetwork {
    active_processors: u64,
    network_regions: Vec<NetworkRegion>,
    consensus_coordinator: ConsensusCoordinator,
    molecular_scheduler: MolecularScheduler,
    oscillation_synchronizer: OscillationSynchronizer,
}

impl MolecularNetwork {
    pub async fn initialize() -> GraffitiResult<Self> {
        tracing::info!("Initializing atmospheric molecular network with 10^44 processors");

        let mut network_regions = Vec::new();
        
        // Create network regions covering different atmospheric layers
        network_regions.push(NetworkRegion::new("troposphere", 0.0, 12000.0).await?);
        network_regions.push(NetworkRegion::new("stratosphere", 12000.0, 50000.0).await?);
        network_regions.push(NetworkRegion::new("mesosphere", 50000.0, 85000.0).await?);
        network_regions.push(NetworkRegion::new("thermosphere", 85000.0, 600000.0).await?);

        let active_processors = atmospheric::TARGET_PROCESSORS_ACTIVE;
        let consensus_coordinator = ConsensusCoordinator::new();
        let molecular_scheduler = MolecularScheduler::new();
        let oscillation_synchronizer = OscillationSynchronizer::new().await?;

        tracing::info!("Atmospheric network initialized with {} active processors", active_processors);

        Ok(Self {
            active_processors,
            network_regions,
            consensus_coordinator,
            molecular_scheduler,
            oscillation_synchronizer,
        })
    }

    pub async fn process_information_molecules(
        &mut self,
        molecules: Vec<InformationMolecule>,
    ) -> GraffitiResult<Vec<InformationMolecule>> {
        tracing::debug!("Processing {} information molecules through atmospheric network", molecules.len());

        // 1. Schedule molecules across the atmospheric network
        let scheduled_tasks = self.molecular_scheduler.schedule_molecules(&molecules).await?;
        
        // 2. Distribute tasks to network regions
        let mut processed_molecules = Vec::new();
        
        for task in scheduled_tasks {
            let region = self.select_optimal_region(&task).await?;
            let result = region.process_task(task).await?;
            processed_molecules.extend(result);
        }

        // 3. Coordinate molecular consensus for validation
        let consensus_score = self.consensus_coordinator
            .coordinate_consensus(&processed_molecules).await?;
        
        if consensus_score < atmospheric::CONSENSUS_THRESHOLD {
            return Err(GraffitiError::AtmosphericProcessing {
                message: format!("Molecular consensus failed: {:.3} < {:.3}", 
                    consensus_score, atmospheric::CONSENSUS_THRESHOLD),
            });
        }

        // 4. Apply oscillation synchronization for temporal coordination
        let synchronized_molecules = self.oscillation_synchronizer
            .synchronize_molecules(processed_molecules).await?;

        tracing::debug!("Atmospheric processing completed with consensus score: {:.3}", consensus_score);
        Ok(synchronized_molecules)
    }

    async fn select_optimal_region(&self, task: &ProcessingTask) -> GraffitiResult<&NetworkRegion> {
        // Select region based on task requirements and current atmospheric conditions
        let optimal_region = match task.task_type {
            TaskType::LogicalProcessing => {
                // Use troposphere for N₂-based logical processing (highest density)
                &self.network_regions[0]
            }
            TaskType::EvidenceValidation => {
                // Use stratosphere for O₂-based evidence validation (stable conditions)
                &self.network_regions[1]
            }
            TaskType::ContextualAnalysis => {
                // Use mesosphere for H₂O-based contextual analysis
                &self.network_regions[2]
            }
            TaskType::EdgeCaseProcessing => {
                // Use thermosphere for trace gas edge case processing
                &self.network_regions[3]
            }
        };

        Ok(optimal_region)
    }

    pub async fn get_network_health(&self) -> GraffitiResult<NetworkHealth> {
        let mut region_health = HashMap::new();
        
        for region in &self.network_regions {
            let health = region.get_health().await?;
            region_health.insert(region.name.clone(), health);
        }

        let overall_health = region_health.values().sum::<f64>() / region_health.len() as f64;

        Ok(NetworkHealth {
            overall_health,
            active_processors: self.active_processors,
            region_health,
            consensus_capability: self.consensus_coordinator.get_capability_score().await?,
            oscillation_sync_quality: self.oscillation_synchronizer.get_sync_quality().await?,
        })
    }
}

/// Represents a region of the atmospheric network
pub struct NetworkRegion {
    pub name: String,
    pub altitude_range: (f64, f64),
    pub active_molecules: u64,
    pub molecular_types: MolecularTypeDistribution,
    pub processing_capacity: f64,
    pub current_load: f64,
}

impl NetworkRegion {
    async fn new(name: &str, min_altitude: f64, max_altitude: f64) -> GraffitiResult<Self> {
        let molecular_types = MolecularTypeDistribution::for_altitude_range(min_altitude, max_altitude);
        let active_molecules = Self::calculate_molecules_in_range(min_altitude, max_altitude);
        
        Ok(Self {
            name: name.to_string(),
            altitude_range: (min_altitude, max_altitude),
            active_molecules,
            molecular_types,
            processing_capacity: (active_molecules as f64 / physics::ATMOSPHERIC_MOLECULES as f64).min(1.0),
            current_load: 0.0,
        })
    }

    fn calculate_molecules_in_range(min_altitude: f64, max_altitude: f64) -> u64 {
        // Calculate atmospheric density based on altitude
        let avg_altitude = (min_altitude + max_altitude) / 2.0;
        let density_factor = (-avg_altitude / 8400.0).exp(); // Scale height approximation
        
        let region_fraction = density_factor * (max_altitude - min_altitude) / 100000.0; // Normalize
        (physics::ATMOSPHERIC_MOLECULES as f64 * region_fraction) as u64
    }

    async fn process_task(&self, task: ProcessingTask) -> GraffitiResult<Vec<InformationMolecule>> {
        tracing::trace!("Region {} processing task of type {:?}", self.name, task.task_type);

        match task.task_type {
            TaskType::LogicalProcessing => self.process_logical_task(task).await,
            TaskType::EvidenceValidation => self.process_evidence_task(task).await,
            TaskType::ContextualAnalysis => self.process_context_task(task).await,
            TaskType::EdgeCaseProcessing => self.process_edge_case_task(task).await,
        }
    }

    async fn process_logical_task(&self, task: ProcessingTask) -> GraffitiResult<Vec<InformationMolecule>> {
        // Use N₂ molecules for logical structure processing
        let n2_processors = (self.active_molecules as f64 * self.molecular_types.n2_fraction) as u64;
        
        let mut processed_molecules = Vec::new();
        for molecule in task.input_molecules {
            let enhanced_molecule = InformationMolecule {
                energy: molecule.energy * 1.1, // N₂ enhances logical energy
                entropy: molecule.entropy * 0.9, // Reduces uncertainty
                temperature: molecule.temperature,
                pressure: molecule.pressure,
                velocity: molecule.velocity,
                content: format!("N2_LOGICAL: {}", molecule.content),
                significance: molecule.significance * 1.2,
            };
            processed_molecules.push(enhanced_molecule);
        }

        tracing::trace!("Logical processing completed with {} N₂ processors", n2_processors);
        Ok(processed_molecules)
    }

    async fn process_evidence_task(&self, task: ProcessingTask) -> GraffitiResult<Vec<InformationMolecule>> {
        // Use O₂ molecules for evidence validation
        let o2_processors = (self.active_molecules as f64 * self.molecular_types.o2_fraction) as u64;
        
        let mut processed_molecules = Vec::new();
        for molecule in task.input_molecules {
            // O₂ molecules validate and strengthen evidence
            let validated_molecule = InformationMolecule {
                energy: molecule.energy,
                entropy: molecule.entropy * 0.8, // O₂ reduces uncertainty through validation
                temperature: molecule.temperature + 10.0, // Validation process adds thermal energy
                pressure: molecule.pressure,
                velocity: molecule.velocity,
                content: format!("O2_VALIDATED: {}", molecule.content),
                significance: if molecule.significance > 0.5 { 
                    molecule.significance * 1.3 // Boost significant evidence
                } else {
                    molecule.significance * 0.7 // Reduce weak evidence
                },
            };
            processed_molecules.push(validated_molecule);
        }

        tracing::trace!("Evidence validation completed with {} O₂ processors", o2_processors);
        Ok(processed_molecules)
    }

    async fn process_context_task(&self, task: ProcessingTask) -> GraffitiResult<Vec<InformationMolecule>> {
        // Use H₂O molecules for contextual coherence
        let h2o_processors = (self.active_molecules as f64 * self.molecular_types.h2o_fraction) as u64;
        
        let mut processed_molecules = Vec::new();
        for molecule in task.input_molecules {
            // H₂O provides contextual coherence through hydrogen bonding
            let contextualized_molecule = InformationMolecule {
                energy: molecule.energy,
                entropy: molecule.entropy,
                temperature: molecule.temperature,
                pressure: molecule.pressure,
                velocity: molecule.velocity,
                content: format!("H2O_CONTEXT: {}", molecule.content),
                significance: molecule.significance, // Context preserves original significance
            };
            processed_molecules.push(contextualized_molecule);
        }

        tracing::trace!("Context analysis completed with {} H₂O processors", h2o_processors);
        Ok(processed_molecules)
    }

    async fn process_edge_case_task(&self, task: ProcessingTask) -> GraffitiResult<Vec<InformationMolecule>> {
        // Use trace gases for edge case processing
        let trace_processors = (self.active_molecules as f64 * self.molecular_types.trace_fraction) as u64;
        
        let mut processed_molecules = Vec::new();
        for molecule in task.input_molecules {
            // Trace gases handle unusual cases
            let edge_processed_molecule = InformationMolecule {
                energy: molecule.energy * 0.8, // Edge cases require less energy
                entropy: molecule.entropy * 1.5, // But have higher uncertainty
                temperature: molecule.temperature,
                pressure: molecule.pressure,
                velocity: molecule.velocity,
                content: format!("TRACE_EDGE: {}", molecule.content),
                significance: molecule.significance * 0.9, // Slightly reduced significance
            };
            processed_molecules.push(edge_processed_molecule);
        }

        tracing::trace!("Edge case processing completed with {} trace gas processors", trace_processors);
        Ok(processed_molecules)
    }

    async fn get_health(&self) -> GraffitiResult<f64> {
        // Calculate region health based on molecular activity and processing capacity
        let load_factor = 1.0 - self.current_load; // Lower load = better health
        let capacity_factor = self.processing_capacity;
        let molecular_factor = (self.active_molecules as f64 / atmospheric::MIN_PROCESSORS_ACTIVE as f64).min(1.0);
        
        let health = (load_factor * 0.4 + capacity_factor * 0.3 + molecular_factor * 0.3).clamp(0.0, 1.0);
        Ok(health)
    }
}

/// Distribution of molecular types in an atmospheric region
#[derive(Debug, Clone)]
pub struct MolecularTypeDistribution {
    pub n2_fraction: f64,
    pub o2_fraction: f64,
    pub h2o_fraction: f64,
    pub trace_fraction: f64,
}

impl MolecularTypeDistribution {
    fn for_altitude_range(min_altitude: f64, _max_altitude: f64) -> Self {
        // Adjust molecular distribution based on altitude
        let altitude_factor = if min_altitude < 15000.0 {
            1.0 // Troposphere - standard distribution
        } else if min_altitude < 50000.0 {
            0.8 // Stratosphere - reduced water vapor
        } else {
            0.6 // Higher altitudes - reduced overall density
        };

        Self {
            n2_fraction: atmospheric::N2_PERCENTAGE / 100.0 * altitude_factor,
            o2_fraction: atmospheric::O2_PERCENTAGE / 100.0 * altitude_factor,
            h2o_fraction: (atmospheric::H2O_MAX_PERCENTAGE / 100.0 * altitude_factor * 0.5).max(0.001),
            trace_fraction: 0.02 * altitude_factor,
        }
    }
}

/// Task for atmospheric molecular processing
#[derive(Debug, Clone)]
pub struct ProcessingTask {
    pub id: uuid::Uuid,
    pub task_type: TaskType,
    pub input_molecules: Vec<InformationMolecule>,
    pub priority: TaskPriority,
    pub deadline: Option<SystemTime>,
}

#[derive(Debug, Clone)]
pub enum TaskType {
    LogicalProcessing,    // N₂-based logical structure processing
    EvidenceValidation,   // O₂-based evidence validation
    ContextualAnalysis,   // H₂O-based contextual coherence
    EdgeCaseProcessing,   // Trace gas-based edge case handling
}

#[derive(Debug, Clone)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Coordinates consensus among atmospheric molecular processors
pub struct ConsensusCoordinator {
    consensus_history: Vec<ConsensusResult>,
    consensus_threshold: f64,
}

impl ConsensusCoordinator {
    fn new() -> Self {
        Self {
            consensus_history: Vec::new(),
            consensus_threshold: atmospheric::CONSENSUS_THRESHOLD,
        }
    }

    async fn coordinate_consensus(&mut self, molecules: &[InformationMolecule]) -> GraffitiResult<f64> {
        if molecules.is_empty() {
            return Ok(0.0);
        }

        // Calculate consensus based on molecular agreement
        let mut agreement_scores = Vec::new();
        
        for molecule in molecules {
            // Calculate how well this molecule agrees with others
            let agreement = self.calculate_molecular_agreement(molecule, molecules).await?;
            agreement_scores.push(agreement);
        }

        let consensus_score = agreement_scores.iter().sum::<f64>() / agreement_scores.len() as f64;

        // Record consensus result
        let consensus_result = ConsensusResult {
            score: consensus_score,
            participant_count: molecules.len() as u64,
            timestamp: SystemTime::now(),
        };
        
        self.consensus_history.push(consensus_result);
        if self.consensus_history.len() > 1000 {
            self.consensus_history.remove(0);
        }

        Ok(consensus_score)
    }

    async fn calculate_molecular_agreement(
        &self,
        target: &InformationMolecule,
        all_molecules: &[InformationMolecule],
    ) -> GraffitiResult<f64> {
        if all_molecules.len() <= 1 {
            return Ok(1.0);
        }

        let mut similarity_scores = Vec::new();

        for other in all_molecules {
            if std::ptr::eq(target, other) {
                continue; // Skip self-comparison
            }

            // Calculate similarity based on molecular properties
            let energy_similarity = 1.0 - (target.energy - other.energy).abs() / (target.energy + other.energy + 1e-10);
            let entropy_similarity = 1.0 - (target.entropy - other.entropy).abs() / (target.entropy + other.entropy + 1e-10);
            let significance_similarity = 1.0 - (target.significance - other.significance).abs();
            
            let overall_similarity = (energy_similarity + entropy_similarity + significance_similarity) / 3.0;
            similarity_scores.push(overall_similarity.clamp(0.0, 1.0));
        }

        let agreement = similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64;
        Ok(agreement)
    }

    async fn get_capability_score(&self) -> GraffitiResult<f64> {
        if self.consensus_history.is_empty() {
            return Ok(0.5);
        }

        // Calculate capability based on recent consensus success rate
        let recent_results = if self.consensus_history.len() > 10 {
            &self.consensus_history[self.consensus_history.len() - 10..]
        } else {
            &self.consensus_history
        };

        let successful_consensus = recent_results.iter()
            .filter(|r| r.score >= self.consensus_threshold)
            .count();
        
        let capability = successful_consensus as f64 / recent_results.len() as f64;
        Ok(capability)
    }
}

#[derive(Debug, Clone)]
struct ConsensusResult {
    score: f64,
    participant_count: u64,
    timestamp: SystemTime,
}

/// Schedules processing tasks across the atmospheric network
pub struct MolecularScheduler {
    task_queue: Vec<ProcessingTask>,
    scheduling_algorithm: SchedulingAlgorithm,
}

impl MolecularScheduler {
    fn new() -> Self {
        Self {
            task_queue: Vec::new(),
            scheduling_algorithm: SchedulingAlgorithm::LoadBalanced,
        }
    }

    async fn schedule_molecules(&mut self, molecules: &[InformationMolecule]) -> GraffitiResult<Vec<ProcessingTask>> {
        let mut tasks = Vec::new();

        for molecule in molecules {
            let task_type = self.determine_task_type(molecule).await?;
            let priority = self.determine_priority(molecule).await?;
            
            let task = ProcessingTask {
                id: uuid::Uuid::new_v4(),
                task_type,
                input_molecules: vec![molecule.clone()],
                priority,
                deadline: None,
            };
            
            tasks.push(task);
        }

        // Apply scheduling algorithm
        match self.scheduling_algorithm {
            SchedulingAlgorithm::LoadBalanced => {
                tasks.sort_by(|a, b| self.compare_by_load_balance(a, b));
            }
            SchedulingAlgorithm::Priority => {
                tasks.sort_by(|a, b| self.compare_by_priority(a, b));
            }
        }

        Ok(tasks)
    }

    async fn determine_task_type(&self, molecule: &InformationMolecule) -> GraffitiResult<TaskType> {
        // Determine processing type based on molecular properties
        if molecule.content.contains("logical") || molecule.content.contains("proof") {
            Ok(TaskType::LogicalProcessing)
        } else if molecule.content.contains("evidence") || molecule.content.contains("validation") {
            Ok(TaskType::EvidenceValidation)
        } else if molecule.content.contains("context") || molecule.content.contains("coherence") {
            Ok(TaskType::ContextualAnalysis)
        } else if molecule.significance < 0.3 {
            Ok(TaskType::EdgeCaseProcessing)
        } else {
            // Default to logical processing
            Ok(TaskType::LogicalProcessing)
        }
    }

    async fn determine_priority(&self, molecule: &InformationMolecule) -> GraffitiResult<TaskPriority> {
        if molecule.significance > 0.9 {
            Ok(TaskPriority::Critical)
        } else if molecule.significance > 0.7 {
            Ok(TaskPriority::High)
        } else if molecule.significance > 0.3 {
            Ok(TaskPriority::Normal)
        } else {
            Ok(TaskPriority::Low)
        }
    }

    fn compare_by_load_balance(&self, a: &ProcessingTask, b: &ProcessingTask) -> std::cmp::Ordering {
        // Simple load balancing - could be enhanced with actual region load data
        a.id.cmp(&b.id)
    }

    fn compare_by_priority(&self, a: &ProcessingTask, b: &ProcessingTask) -> std::cmp::Ordering {
        let a_priority_value = match a.priority {
            TaskPriority::Critical => 4,
            TaskPriority::High => 3,
            TaskPriority::Normal => 2,
            TaskPriority::Low => 1,
        };
        let b_priority_value = match b.priority {
            TaskPriority::Critical => 4,
            TaskPriority::High => 3,
            TaskPriority::Normal => 2,
            TaskPriority::Low => 1,
        };
        
        b_priority_value.cmp(&a_priority_value)
    }
}

#[derive(Debug, Clone)]
enum SchedulingAlgorithm {
    LoadBalanced,
    Priority,
}

/// Synchronizes molecular oscillations for temporal coordination
pub struct OscillationSynchronizer {
    n2_frequency: f64,
    o2_frequency: f64,
    sync_quality: f64,
}

impl OscillationSynchronizer {
    async fn new() -> GraffitiResult<Self> {
        Ok(Self {
            n2_frequency: atmospheric::N2_VIBRATIONAL_FREQUENCY,
            o2_frequency: atmospheric::O2_VIBRATIONAL_FREQUENCY,
            sync_quality: 0.95,
        })
    }

    async fn synchronize_molecules(
        &mut self,
        molecules: Vec<InformationMolecule>,
    ) -> GraffitiResult<Vec<InformationMolecule>> {
        let mut synchronized = Vec::new();
        
        for molecule in molecules {
            let sync_molecule = self.apply_oscillation_sync(molecule).await?;
            synchronized.push(sync_molecule);
        }

        Ok(synchronized)
    }

    async fn apply_oscillation_sync(&self, mut molecule: InformationMolecule) -> GraffitiResult<InformationMolecule> {
        // Apply molecular oscillation synchronization for temporal coordination
        let current_time = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as f64;

        // Calculate phase alignment with N₂ or O₂ oscillations
        let phase_alignment = if molecule.content.contains("N2") {
            (current_time * self.n2_frequency * 1e-9) % (2.0 * std::f64::consts::PI)
        } else if molecule.content.contains("O2") {
            (current_time * self.o2_frequency * 1e-9) % (2.0 * std::f64::consts::PI)
        } else {
            0.0
        };

        // Apply phase correction to molecular velocity for temporal coordination
        let phase_correction = phase_alignment.sin() * 0.1;
        molecule.velocity = Vector3::new(
            molecule.velocity.x * (1.0 + phase_correction),
            molecule.velocity.y * (1.0 + phase_correction),
            molecule.velocity.z * (1.0 + phase_correction),
        );

        Ok(molecule)
    }

    async fn get_sync_quality(&self) -> GraffitiResult<f64> {
        Ok(self.sync_quality)
    }
}

/// Health status of the atmospheric molecular network
#[derive(Debug, Clone)]
pub struct NetworkHealth {
    pub overall_health: f64,
    pub active_processors: u64,
    pub region_health: HashMap<String, f64>,
    pub consensus_capability: f64,
    pub oscillation_sync_quality: f64,
}
