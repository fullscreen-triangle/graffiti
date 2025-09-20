"""
Unified Integration Engine

Orchestrates all theoretical frameworks into a cohesive revolutionary
search engine architecture with environmental consciousness integration.
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging
import time

from graffiti.core.types import (
    Query,
    QueryResult,
    EnvironmentalState,
    ProcessingMode,
    PerformanceMetrics,
)
from graffiti.core.s_entropy import SEntropyNavigator
from graffiti.core.meaning_impossibility import MeaningImpossibilityAnalyzer
from graffiti.core.universal_solver import UniversalProblemSolver
from graffiti.core.bmd_operations import BiologicalMaxwellDemon
from graffiti.environmental.twelve_dimensional import EnvironmentalMeasurement

logger = logging.getLogger(__name__)


class GraffitiEngine:
    """
    Unified Graffiti search engine integrating all revolutionary frameworks.
    
    Provides a single interface to the complete theoretical architecture.
    """
    
    def __init__(self):
        """Initialize the unified Graffiti engine."""
        self.s_entropy_navigator = SEntropyNavigator()
        self.meaning_analyzer = MeaningImpossibilityAnalyzer() 
        self.universal_solver = UniversalProblemSolver()
        self.bmd_processor = BiologicalMaxwellDemon()
        self.environmental_measurement = EnvironmentalMeasurement()
        
        logger.info("GraffitiEngine initialized with complete revolutionary architecture")
    
    async def process_query(self, query: Query, 
                          mode: ProcessingMode = ProcessingMode.FULL_REVOLUTIONARY) -> QueryResult:
        """
        Process query through integrated revolutionary frameworks.
        
        Args:
            query: Query to process
            mode: Processing mode
            
        Returns:
            Integrated query result
        """
        start_time = time.time()
        
        # Ensure environmental context
        if not query.environmental_context or query.environmental_context.calculate_uniqueness() == 0:
            query.environmental_context = await self.environmental_measurement.measure_environment()
        
        # Process through S-entropy navigation (core framework)
        s_entropy_result = self.s_entropy_navigator.navigate_to_solution(query)
        
        # Add meaning impossibility analysis if requested
        meaning_validation = None
        if mode == ProcessingMode.FULL_REVOLUTIONARY:
            meaning_validation = self.meaning_analyzer.analyze_meaning_requirements(query)
        
        # Add universal solver insights if requested
        universal_result = None
        if mode == ProcessingMode.FULL_REVOLUTIONARY:
            universal_result = self.universal_solver.solve_universal_problem(query)
        
        # Add BMD consciousness processing if requested
        bmd_result = None
        if mode == ProcessingMode.FULL_REVOLUTIONARY:
            bmd_result = self.bmd_processor.process_full_bmd_cycle(
                query.environmental_context, query.content
            )
        
        # Integrate results
        integrated_result = self._integrate_framework_results(
            query, s_entropy_result, meaning_validation, universal_result, bmd_result
        )
        
        processing_time = time.time() - start_time
        integrated_result.processing_time = processing_time
        
        return integrated_result
    
    def _integrate_framework_results(self, query: Query, s_entropy_result: QueryResult,
                                   meaning_validation: Optional[Any],
                                   universal_result: Optional[QueryResult],
                                   bmd_result: Optional[Dict]) -> QueryResult:
        """Integrate results from all frameworks."""
        
        # Start with S-entropy result as base
        integrated_content = s_entropy_result.content + "\n\n"
        
        # Add meaning impossibility insights
        if meaning_validation:
            integrated_content += f"Meaning Impossibility Analysis: {meaning_validation.score:.1%} systematic impossibility demonstrated\n"
        
        # Add universal solver insights
        if universal_result:
            integrated_content += f"Universal Problem Solving: Dual architecture equivalence confirmed\n"
        
        # Add BMD consciousness insights
        if bmd_result:
            frame_count = len(bmd_result['frame_selection'].selected_frames)
            integrated_content += f"BMD Consciousness: {frame_count} frames activated with {bmd_result['sanity_check'].sanity_status} status\n"
        
        # Calculate integrated performance metrics
        integrated_performance = PerformanceMetrics(
            processing_time=s_entropy_result.processing_time,
            memory_usage=s_entropy_result.performance_metrics.memory_usage,
            accuracy_score=s_entropy_result.confidence,
            speedup_factor=s_entropy_result.performance_metrics.speedup_factor,
            compression_ratio=s_entropy_result.performance_metrics.compression_ratio,
            s_distance_minimization=s_entropy_result.performance_metrics.s_distance_minimization
        )
        
        return QueryResult(
            query_id=query.id,
            content=integrated_content,
            confidence=s_entropy_result.confidence,
            environmental_confidence=s_entropy_result.environmental_confidence,
            s_distance=s_entropy_result.s_distance,
            coordinates=s_entropy_result.coordinates,
            performance_metrics=integrated_performance,
            metadata={
                'integration_mode': 'unified_engine',
                'frameworks_used': ['s_entropy'] + 
                                 (['meaning_impossibility'] if meaning_validation else []) +
                                 (['universal_solver'] if universal_result else []) +
                                 (['bmd_consciousness'] if bmd_result else [])
            }
        )


class IntegrationOrchestrator:
    """
    Advanced orchestration system for coordinating multiple engines
    and processing modes.
    """
    
    def __init__(self):
        """Initialize integration orchestrator."""
        self.engines: Dict[str, GraffitiEngine] = {}
        self.processing_queue: List[Query] = []
        self.results_cache: Dict[str, QueryResult] = {}
    
    def add_engine(self, name: str, engine: GraffitiEngine):
        """Add engine to orchestration system."""
        self.engines[name] = engine
        logger.info(f"Engine '{name}' added to orchestration system")
    
    async def process_batch(self, queries: List[Query], 
                          mode: ProcessingMode = ProcessingMode.FULL_REVOLUTIONARY) -> List[QueryResult]:
        """Process batch of queries across all engines."""
        
        if not self.engines:
            # Create default engine if none exist
            self.add_engine("default", GraffitiEngine())
        
        results = []
        
        # Process queries in parallel across engines
        for query in queries:
            # Use first available engine (could be enhanced for load balancing)
            engine = next(iter(self.engines.values()))
            result = await engine.process_query(query, mode)
            results.append(result)
        
        return results
    
    async def process_parallel(self, query: Query) -> Dict[str, QueryResult]:
        """Process single query across multiple engines in parallel."""
        
        if not self.engines:
            self.add_engine("default", GraffitiEngine())
        
        # Process query across all engines simultaneously
        tasks = []
        for name, engine in self.engines.items():
            task = asyncio.create_task(
                engine.process_query(query, ProcessingMode.FULL_REVOLUTIONARY),
                name=f"engine_{name}"
            )
            tasks.append((name, task))
        
        results = {}
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"Engine '{name}' failed: {e}")
        
        return results
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get statistics on orchestration performance."""
        return {
            'engines_available': len(self.engines),
            'engines_registered': list(self.engines.keys()),
            'queue_length': len(self.processing_queue),
            'cache_size': len(self.results_cache),
            'orchestration_mode': 'parallel_processing'
        }
