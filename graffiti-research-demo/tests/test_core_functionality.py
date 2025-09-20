"""
Basic tests for core Graffiti functionality.

Tests the fundamental theoretical framework implementations.
"""

import pytest
import asyncio
from graffiti.core.types import (
    Query, 
    QueryId, 
    EnvironmentalState, 
    SEntropyCoordinates,
    ProcessingMode
)
from graffiti.core.s_entropy import SEntropyCalculator, SEntropyNavigator
from graffiti.core.meaning_impossibility import MeaningImpossibilityAnalyzer
from graffiti.applications.search_demo import GraffitiSearchDemo


class TestSEntropyFramework:
    """Test S-entropy framework implementation."""
    
    def test_s_entropy_calculator(self):
        """Test basic S-entropy calculation."""
        calculator = SEntropyCalculator()
        
        # Test basic calculation: S = k * log(α)
        result = calculator.calculate_s_value(2.0)
        assert result > 0, "S-entropy should be positive for amplitude > 1"
        
        result = calculator.calculate_s_value(0.5)
        assert result < 0, "S-entropy should be negative for amplitude < 1"
    
    def test_coordinate_conversion(self):
        """Test coordinate conversion functions."""
        calculator = SEntropyCalculator()
        
        # Test amplitude to coordinates
        coords = calculator.amplitude_to_coordinates(1.0)
        assert isinstance(coords, SEntropyCoordinates)
        assert 0 <= coords.knowledge <= 1
        assert 0 <= coords.time <= 1
        assert 0 <= coords.entropy <= 1
        
        # Test coordinates to amplitude
        amplitude = calculator.coordinates_to_amplitude(coords)
        assert amplitude > 0, "Amplitude should be positive"
    
    @pytest.mark.asyncio
    async def test_s_entropy_navigation(self):
        """Test S-entropy navigation system."""
        navigator = SEntropyNavigator()
        
        query = Query(
            content="test optimization query",
            environmental_context=EnvironmentalState()
        )
        
        result = await navigator.navigate_to_solution(query)
        
        assert result.query_id == query.id
        assert result.confidence > 0
        assert result.s_distance >= 0
        assert "S-entropy navigation" in result.content


class TestMeaningImpossibilityFramework:
    """Test meaning impossibility analysis."""
    
    def test_meaning_impossibility_analyzer(self):
        """Test meaning impossibility analysis."""
        analyzer = MeaningImpossibilityAnalyzer()
        
        query = Query(content="What is the meaning of existence?")
        
        result = analyzer.analyze_meaning_requirements(query)
        
        assert result.test_name == "meaning_impossibility_analysis"
        assert 0 <= result.score <= 1
        assert result.score > 0.5, "Should demonstrate high impossibility"
    
    def test_impossibility_proof_retrieval(self):
        """Test impossibility proof retrieval."""
        analyzer = MeaningImpossibilityAnalyzer()
        
        from graffiti.core.meaning_impossibility import ImpossibilityType
        
        proof = analyzer.get_impossibility_proof(ImpossibilityType.TEMPORAL_PREDETERMINATION)
        assert proof is not None
        assert proof.impossibility_factor > 0.9, "Should show high impossibility"


class TestEnvironmentalMeasurement:
    """Test environmental measurement system."""
    
    @pytest.mark.asyncio
    async def test_environmental_measurement(self):
        """Test basic environmental measurement."""
        from graffiti.environmental.twelve_dimensional import EnvironmentalMeasurement
        
        measurement_system = EnvironmentalMeasurement()
        env_state = await measurement_system.measure_environment()
        
        assert isinstance(env_state, EnvironmentalState)
        assert env_state.calculate_uniqueness() >= 0
        assert env_state.quantum_coherence >= 0
        assert env_state.temporal_flow > 0


class TestSearchDemo:
    """Test main search demo application."""
    
    @pytest.mark.asyncio
    async def test_basic_search(self):
        """Test basic search functionality."""
        demo = GraffitiSearchDemo()
        
        result = await demo.search(
            "pattern recognition optimization", 
            ProcessingMode.S_ENTROPY_NAVIGATION
        )
        
        assert 'query' in result
        assert 'environmental_context' in result
        assert 'framework_results' in result
        assert 'synthesis' in result
        
        # Check that S-entropy framework was used
        assert 's_entropy' in result['framework_results']
        
        # Verify performance metrics
        assert result['performance']['total_processing_time'] > 0
    
    @pytest.mark.asyncio 
    async def test_full_revolutionary_search(self):
        """Test search with full revolutionary processing."""
        demo = GraffitiSearchDemo()
        
        result = await demo.search(
            "consciousness and meaning",
            ProcessingMode.FULL_REVOLUTIONARY
        )
        
        frameworks = result['framework_results']
        
        # Should include all frameworks
        assert 's_entropy' in frameworks
        assert 'meaning_impossibility' in frameworks
        assert 'universal_solver' in frameworks
        assert 'bmd_consciousness' in frameworks
        
        # Check synthesis quality
        synthesis = result['synthesis']
        assert len(synthesis['revolutionary_insights']) > 0
        assert synthesis['final_answer'] != ""
        assert 'consciousness_integration' in synthesis
    
    def test_search_statistics(self):
        """Test search statistics calculation."""
        demo = GraffitiSearchDemo()
        
        # Should handle empty history
        stats = demo.get_search_statistics()
        assert 'message' in stats or 'total_searches' in stats


class TestIntegration:
    """Test framework integration."""
    
    @pytest.mark.asyncio
    async def test_unified_engine(self):
        """Test unified engine integration."""
        from graffiti.integration.unified_engine import GraffitiEngine
        
        engine = GraffitiEngine()
        
        query = Query(
            content="test integration query",
            environmental_context=EnvironmentalState()
        )
        
        result = await engine.process_query(query, ProcessingMode.FULL_REVOLUTIONARY)
        
        assert isinstance(result.query_id, QueryId)
        assert result.confidence > 0
        assert result.s_distance >= 0
        assert len(result.metadata['frameworks_used']) > 1


# Performance benchmarks
@pytest.mark.slow
class TestPerformance:
    """Performance validation tests."""
    
    @pytest.mark.asyncio
    async def test_processing_speed(self):
        """Test that processing completes within reasonable time."""
        demo = GraffitiSearchDemo()
        
        import time
        start_time = time.time()
        
        await demo.search(
            "speed test query",
            ProcessingMode.S_ENTROPY_NAVIGATION
        )
        
        processing_time = time.time() - start_time
        assert processing_time < 5.0, "Processing should complete within 5 seconds"
    
    @pytest.mark.asyncio
    async def test_speedup_claims(self):
        """Test speedup factor claims."""
        demo = GraffitiSearchDemo()
        
        result = await demo.search(
            "optimization problem with multiple constraints",
            ProcessingMode.S_ENTROPY_NAVIGATION
        )
        
        if 's_entropy' in result['framework_results']:
            speedup = result['framework_results']['s_entropy']['performance'].speedup_factor
            assert speedup > 1.0, "Should demonstrate speedup over traditional methods"


if __name__ == "__main__":
    # Run basic tests without pytest
    import sys
    
    async def run_basic_tests():
        """Run basic tests manually."""
        print("Running basic Graffiti functionality tests...")
        
        # Test S-entropy
        print("Testing S-entropy framework...")
        test_s_entropy = TestSEntropyFramework()
        test_s_entropy.test_s_entropy_calculator()
        test_s_entropy.test_coordinate_conversion()
        await test_s_entropy.test_s_entropy_navigation()
        print("✓ S-entropy tests passed")
        
        # Test meaning impossibility
        print("Testing meaning impossibility framework...")
        test_meaning = TestMeaningImpossibilityFramework()
        test_meaning.test_meaning_impossibility_analyzer()
        test_meaning.test_impossibility_proof_retrieval()
        print("✓ Meaning impossibility tests passed")
        
        # Test search demo
        print("Testing search demo...")
        test_demo = TestSearchDemo()
        await test_demo.test_basic_search()
        print("✓ Search demo tests passed")
        
        print("All basic tests completed successfully!")
    
    asyncio.run(run_basic_tests())
