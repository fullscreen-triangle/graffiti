//! Query to information molecule conversion system

use graffiti_core::*;
use nalgebra::Vector3;
use std::time::SystemTime;

/// Converts queries into information molecules for atmospheric processing
pub struct QueryToMoleculeConverter {
    conversion_templates: Vec<ConversionTemplate>,
    environmental_enhancer: EnvironmentalEnhancer,
}

impl QueryToMoleculeConverter {
    pub fn new() -> Self {
        let conversion_templates = vec![
            ConversionTemplate::new("proof", MoleculeType::Logical),
            ConversionTemplate::new("evidence", MoleculeType::Validation),
            ConversionTemplate::new("context", MoleculeType::Contextual),
            ConversionTemplate::new("why", MoleculeType::Causal),
            ConversionTemplate::new("how", MoleculeType::Procedural),
            ConversionTemplate::new("what", MoleculeType::Definitional),
        ];

        Self {
            conversion_templates,
            environmental_enhancer: EnvironmentalEnhancer::new(),
        }
    }

    pub async fn convert_query_to_molecules(
        &self,
        query: &Query,
        env: &EnvironmentalState,
    ) -> GraffitiResult<Vec<InformationMolecule>> {
        tracing::debug!("Converting query to information molecules: {}", query.content);

        // 1. Analyze query structure and intent
        let query_analysis = self.analyze_query_structure(&query.content).await?;
        
        // 2. Create base information molecules from query components
        let mut base_molecules = Vec::new();
        
        // Create molecules for each significant word/phrase
        let query_tokens = self.tokenize_query(&query.content).await?;
        for token in query_tokens {
            let molecule_type = self.determine_molecule_type(&token, &query_analysis).await?;
            let base_molecule = self.create_base_molecule(&token, molecule_type).await?;
            base_molecules.push(base_molecule);
        }

        // 3. Enhance molecules with environmental information
        let enhanced_molecules = self.environmental_enhancer
            .enhance_with_environment(base_molecules, env).await?;

        // 4. Apply query context and user preferences
        let contextualized_molecules = self.apply_query_context(enhanced_molecules, query).await?;

        tracing::debug!("Converted query into {} information molecules", contextualized_molecules.len());
        Ok(contextualized_molecules)
    }

    async fn analyze_query_structure(&self, query: &str) -> GraffitiResult<QueryAnalysis> {
        let query_lower = query.to_lowercase();
        
        // Detect query type
        let query_type = if query_lower.starts_with("why") {
            QueryType::Causal
        } else if query_lower.starts_with("how") {
            QueryType::Procedural
        } else if query_lower.starts_with("what") {
            QueryType::Definitional
        } else if query_lower.contains("prove") || query_lower.contains("proof") {
            QueryType::Proof
        } else if query_lower.contains("explain") {
            QueryType::Explanatory
        } else {
            QueryType::General
        };

        // Detect complexity based on query length and structure
        let complexity = if query.len() > 100 {
            QueryComplexity::High
        } else if query.len() > 50 {
            QueryComplexity::Medium
        } else {
            QueryComplexity::Low
        };

        // Detect domain based on keywords
        let domain = self.detect_domain(&query_lower).await?;

        Ok(QueryAnalysis {
            query_type,
            complexity,
            domain,
            key_concepts: self.extract_key_concepts(&query_lower).await?,
            molecular_requirements: self.estimate_molecular_requirements(&query_type, &complexity).await?,
        })
    }

    async fn detect_domain(&self, query: &str) -> GraffitiResult<QueryDomain> {
        if query.contains("math") || query.contains("theorem") || query.contains("equation") {
            Ok(QueryDomain::Mathematics)
        } else if query.contains("physics") || query.contains("quantum") || query.contains("energy") {
            Ok(QueryDomain::Physics)
        } else if query.contains("computer") || query.contains("algorithm") || query.contains("programming") {
            Ok(QueryDomain::Computer)
        } else if query.contains("biology") || query.contains("cell") || query.contains("organism") {
            Ok(QueryDomain::Biology)
        } else if query.contains("chemistry") || query.contains("molecule") || query.contains("reaction") {
            Ok(QueryDomain::Chemistry)
        } else {
            Ok(QueryDomain::General)
        }
    }

    async fn extract_key_concepts(&self, query: &str) -> GraffitiResult<Vec<String>> {
        // Simple keyword extraction - in a real system this would use NLP
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut concepts = Vec::new();

        for word in words {
            if word.len() > 3 && !self.is_stop_word(word) {
                concepts.push(word.to_string());
            }
        }

        Ok(concepts)
    }

    fn is_stop_word(&self, word: &str) -> bool {
        matches!(word, "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by")
    }

    async fn estimate_molecular_requirements(
        &self,
        query_type: &QueryType,
        complexity: &QueryComplexity,
    ) -> GraffitiResult<MolecularRequirements> {
        let base_molecules = match complexity {
            QueryComplexity::Low => 5,
            QueryComplexity::Medium => 15,
            QueryComplexity::High => 50,
        };

        let processing_intensity = match query_type {
            QueryType::Proof => 1.5,
            QueryType::Causal => 1.3,
            QueryType::Procedural => 1.2,
            QueryType::Explanatory => 1.1,
            _ => 1.0,
        };

        Ok(MolecularRequirements {
            estimated_molecules: (base_molecules as f64 * processing_intensity) as usize,
            processing_depth: match complexity {
                QueryComplexity::Low => 1,
                QueryComplexity::Medium => 2,
                QueryComplexity::High => 3,
            },
            atmospheric_regions_needed: match query_type {
                QueryType::Proof => vec!["troposphere".to_string(), "stratosphere".to_string()],
                QueryType::Causal => vec!["troposphere".to_string()],
                _ => vec!["troposphere".to_string()],
            },
        })
    }

    async fn tokenize_query(&self, query: &str) -> GraffitiResult<Vec<String>> {
        // Simple tokenization - split by whitespace and punctuation
        let tokens: Vec<String> = query
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .map(|word| word.trim_matches(|c: char| c.is_ascii_punctuation()).to_lowercase())
            .filter(|word| !word.is_empty() && !self.is_stop_word(word))
            .collect();

        Ok(tokens)
    }

    async fn determine_molecule_type(
        &self,
        token: &str,
        analysis: &QueryAnalysis,
    ) -> GraffitiResult<MoleculeType> {
        // Determine molecule type based on token and query analysis
        for template in &self.conversion_templates {
            if template.matches_token(token) {
                return Ok(template.molecule_type.clone());
            }
        }

        // Default based on query type
        Ok(match analysis.query_type {
            QueryType::Proof => MoleculeType::Logical,
            QueryType::Causal => MoleculeType::Causal,
            QueryType::Procedural => MoleculeType::Procedural,
            QueryType::Definitional => MoleculeType::Definitional,
            QueryType::Explanatory => MoleculeType::Contextual,
            QueryType::General => MoleculeType::General,
        })
    }

    async fn create_base_molecule(&self, token: &str, molecule_type: MoleculeType) -> GraffitiResult<InformationMolecule> {
        // Create base information molecule from token
        let energy = self.calculate_token_energy(token, &molecule_type).await?;
        let entropy = self.calculate_token_entropy(token).await?;
        let significance = self.calculate_token_significance(token, &molecule_type).await?;

        Ok(InformationMolecule {
            energy,
            entropy,
            temperature: 298.15, // Room temperature baseline
            pressure: physics::ATMOSPHERIC_PRESSURE_SEA_LEVEL,
            velocity: Vector3::new(0.0, 0.0, 0.0), // Initially at rest
            content: token.to_string(),
            significance,
        })
    }

    async fn calculate_token_energy(&self, token: &str, molecule_type: &MoleculeType) -> GraffitiResult<f64> {
        // Calculate information energy based on token importance and type
        let base_energy = token.len() as f64 * 0.1; // Length-based energy
        
        let type_multiplier = match molecule_type {
            MoleculeType::Logical => 1.5,
            MoleculeType::Validation => 1.3,
            MoleculeType::Causal => 1.4,
            MoleculeType::Procedural => 1.2,
            MoleculeType::Definitional => 1.1,
            MoleculeType::Contextual => 1.0,
            MoleculeType::General => 0.8,
        };

        Ok(base_energy * type_multiplier)
    }

    async fn calculate_token_entropy(&self, token: &str) -> GraffitiResult<f64> {
        // Calculate information entropy based on token characteristics
        let char_entropy = self.calculate_character_entropy(token);
        let length_entropy = (token.len() as f64).ln() * 0.1;
        
        Ok(char_entropy + length_entropy)
    }

    fn calculate_character_entropy(&self, token: &str) -> f64 {
        if token.is_empty() {
            return 0.0;
        }

        let mut char_counts = std::collections::HashMap::new();
        for ch in token.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }

        let len = token.len() as f64;
        let mut entropy = 0.0;

        for &count in char_counts.values() {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }

        entropy
    }

    async fn calculate_token_significance(&self, token: &str, molecule_type: &MoleculeType) -> GraffitiResult<f64> {
        // Calculate significance based on token characteristics and type
        let mut significance = 0.5; // Base significance

        // Increase significance for longer tokens (more information)
        significance += (token.len() as f64 * 0.02).min(0.3);

        // Adjust based on molecule type
        significance *= match molecule_type {
            MoleculeType::Logical => 1.2,
            MoleculeType::Validation => 1.1,
            MoleculeType::Causal => 1.15,
            _ => 1.0,
        };

        // Increase significance for domain-specific terms
        if self.is_technical_term(token) {
            significance += 0.2;
        }

        Ok(significance.clamp(0.0, 1.0))
    }

    fn is_technical_term(&self, token: &str) -> bool {
        matches!(token, "algorithm" | "theorem" | "proof" | "quantum" | "molecular" | "entropy" | "energy" | "system")
    }

    async fn apply_query_context(
        &self,
        molecules: Vec<InformationMolecule>,
        query: &Query,
    ) -> GraffitiResult<Vec<InformationMolecule>> {
        let mut contextualized = Vec::new();

        for mut molecule in molecules {
            // Apply user expertise level
            match query.user_context.expertise_level {
                ExpertiseLevel::Beginner => {
                    molecule.entropy *= 0.8; // Reduce uncertainty for beginners
                    molecule.significance *= 1.1; // Boost significance
                }
                ExpertiseLevel::Expert => {
                    molecule.entropy *= 1.2; // Allow more uncertainty for experts
                    molecule.energy *= 1.1; // Increase processing energy
                }
                _ => {} // No adjustment for intermediate levels
            }

            // Apply query urgency
            match query.urgency {
                Urgency::High | Urgency::Critical => {
                    molecule.energy *= 1.3; // Boost energy for urgent queries
                    molecule.velocity = Vector3::new(1.0, 1.0, 1.0); // Increase processing velocity
                }
                _ => {}
            }

            contextualized.push(molecule);
        }

        Ok(contextualized)
    }
}

/// Enhances information molecules with environmental information
pub struct EnvironmentalEnhancer;

impl EnvironmentalEnhancer {
    fn new() -> Self {
        Self
    }

    async fn enhance_with_environment(
        &self,
        molecules: Vec<InformationMolecule>,
        env: &EnvironmentalState,
    ) -> GraffitiResult<Vec<InformationMolecule>> {
        let mut enhanced = Vec::new();

        for mut molecule in molecules {
            // Apply atmospheric conditions
            molecule.pressure = env.atmospheric.pressure;
            molecule.temperature = env.atmospheric.temperature;

            // Apply temporal precision enhancement
            if env.temporal.precision_by_difference > temporal::TARGET_PRECISION {
                molecule.energy *= 1.1; // Boost energy with high temporal precision
            }

            // Apply quantum coherence enhancement
            if env.quantum.quantum_coherence > 0.7 {
                molecule.entropy *= (1.0 - env.quantum.quantum_coherence * 0.2); // Reduce entropy
                molecule.significance *= (1.0 + env.quantum.quantum_coherence * 0.1); // Boost significance
            }

            // Apply biometric state influence
            if env.biometric.cognitive_load > 0.8 {
                molecule.energy *= 1.2; // High cognitive load increases processing energy
            }

            // Apply spatial/gravitational influence
            let gravity_factor = env.spatial.gravitational_field / physics::EARTH_GRAVITY;
            molecule.velocity = Vector3::new(
                molecule.velocity.x * gravity_factor,
                molecule.velocity.y * gravity_factor,
                molecule.velocity.z * gravity_factor,
            );

            enhanced.push(molecule);
        }

        Ok(enhanced)
    }
}

/// Template for converting query elements to molecules
#[derive(Debug, Clone)]
pub struct ConversionTemplate {
    pub keyword: String,
    pub molecule_type: MoleculeType,
}

impl ConversionTemplate {
    fn new(keyword: &str, molecule_type: MoleculeType) -> Self {
        Self {
            keyword: keyword.to_string(),
            molecule_type,
        }
    }

    fn matches_token(&self, token: &str) -> bool {
        token.contains(&self.keyword)
    }
}

/// Type of information molecule based on its role
#[derive(Debug, Clone)]
pub enum MoleculeType {
    Logical,        // For logical reasoning and proofs
    Validation,     // For evidence validation
    Contextual,     // For context and coherence
    Causal,         // For causal relationships
    Procedural,     // For procedures and processes
    Definitional,   // For definitions and descriptions
    General,        // General information processing
}

/// Analysis of query structure and requirements
#[derive(Debug)]
struct QueryAnalysis {
    query_type: QueryType,
    complexity: QueryComplexity,
    domain: QueryDomain,
    key_concepts: Vec<String>,
    molecular_requirements: MolecularRequirements,
}

#[derive(Debug)]
enum QueryType {
    Proof,
    Causal,
    Procedural,
    Definitional,
    Explanatory,
    General,
}

#[derive(Debug)]
enum QueryComplexity {
    Low,
    Medium,
    High,
}

#[derive(Debug)]
enum QueryDomain {
    Mathematics,
    Physics,
    Computer,
    Biology,
    Chemistry,
    General,
}

#[derive(Debug)]
struct MolecularRequirements {
    estimated_molecules: usize,
    processing_depth: u32,
    atmospheric_regions_needed: Vec<String>,
}
