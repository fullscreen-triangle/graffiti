//! Graffiti Atmospheric Processing Network
//!
//! Harnesses Earth's 10^44 atmospheric molecules as distributed processors
//! for revolutionary proof construction and validation.

use graffiti_core::*;
use async_trait::async_trait;
use nalgebra::Vector3;
use std::time::SystemTime;
use uuid::Uuid;

pub mod network;
pub mod conversion;

use network::MolecularNetwork;
use conversion::QueryToMoleculeConverter;

/// Main atmospheric processing network that coordinates Earth's molecular infrastructure
pub struct AtmosphericProcessingNetwork {
    molecular_network: MolecularNetwork,
    query_converter: QueryToMoleculeConverter,
}

impl AtmosphericProcessingNetwork {
    pub async fn new() -> GraffitiResult<Self> {
        tracing::info!("Initializing atmospheric processing network with Earth's 10^44 molecules");
        
        let molecular_network = MolecularNetwork::initialize().await?;
        let query_converter = QueryToMoleculeConverter::new();
        
        tracing::info!("Atmospheric processing network ready for distributed molecular computation");
        
        Ok(Self {
            molecular_network,
            query_converter,
        })
    }
    
    pub async fn process_query(
        &mut self,
        query: &Query,
        env: &EnvironmentalState,
    ) -> GraffitiResult<Vec<InformationMolecule>> {
        tracing::debug!("Processing query through atmospheric molecular network: {}", query.content);
        
        // Convert query to information molecules using environmental context
        let query_molecules = self.query_converter
            .convert_query_to_molecules(query, env).await?;
        
        // Process through atmospheric molecular network
        let processed_molecules = self.molecular_network
            .process_information_molecules(query_molecules).await?;
        
        tracing::debug!("Query processing completed with {} result molecules", processed_molecules.len());
        Ok(processed_molecules)
    }
    
    pub async fn health_check(&self) -> GraffitiResult<ComponentStatus> {
        let network_health = self.molecular_network.get_network_health().await?;
        
        if network_health.overall_health > 0.8 {
            Ok(ComponentStatus::Healthy)
        } else if network_health.overall_health > 0.6 {
            Ok(ComponentStatus::Degraded {
                reason: format!("Network health degraded: {:.3}", network_health.overall_health),
            })
        } else {
            Ok(ComponentStatus::Unhealthy {
                reason: format!("Network health critical: {:.3}", network_health.overall_health),
            })
        }
    }
}

#[async_trait]
impl AtmosphericProcessing for AtmosphericProcessingNetwork {
    async fn process_atmospheric(
        &mut self,
        information: Vec<InformationMolecule>,
    ) -> GraffitiResult<Vec<InformationMolecule>> {
        self.molecular_network.process_information_molecules(information).await
    }
    
    async fn get_processor_count(&self) -> GraffitiResult<u64> {
        let health = self.molecular_network.get_network_health().await?;
        Ok(health.active_processors)
    }
    
    async fn network_health(&self) -> GraffitiResult<f64> {
        let health = self.molecular_network.get_network_health().await?;
        Ok(health.overall_health)
    }
    
    async fn coordinate_consensus(
        &mut self,
        molecules: &[InformationMolecule],
    ) -> GraffitiResult<f64> {
        // Process molecules to get consensus score
        let processed = self.molecular_network
            .process_information_molecules(molecules.to_vec()).await?;
        
        // Calculate consensus based on molecular agreement
        if processed.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_significance = 0.0;
        let mut count = 0;
        
        for molecule in &processed {
            total_significance += molecule.significance;
            count += 1;
        }
        
        let consensus_score = if count > 0 {
            total_significance / count as f64
        } else {
            0.0
        };
        
        Ok(consensus_score.clamp(0.0, 1.0))
    }
}
