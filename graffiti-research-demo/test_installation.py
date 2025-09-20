#!/usr/bin/env python3
"""
Quick installation test for Graffiti Research Demo package.

Tests basic imports and functionality to ensure the package is working correctly.
"""

import sys
import traceback
import asyncio


def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic imports...")
    
    try:
        import graffiti
        print("✓ Main package import successful")
    except Exception as e:
        print(f"✗ Main package import failed: {e}")
        return False
    
    try:
        from graffiti.core.types import Query, EnvironmentalState, SEntropyCoordinates
        print("✓ Core types import successful")
    except Exception as e:
        print(f"✗ Core types import failed: {e}")
        return False
    
    try:
        from graffiti.core.s_entropy import SEntropyCalculator
        print("✓ S-entropy framework import successful")
    except Exception as e:
        print(f"✗ S-entropy framework import failed: {e}")
        return False
    
    try:
        from graffiti.applications.search_demo import GraffitiSearchDemo
        print("✓ Search demo import successful")
    except Exception as e:
        print(f"✗ Search demo import failed: {e}")
        return False
    
    return True


def test_s_entropy_basic_functionality():
    """Test basic S-entropy functionality."""
    print("\nTesting S-entropy basic functionality...")
    
    try:
        from graffiti.core.s_entropy import SEntropyCalculator
        calculator = SEntropyCalculator()
        
        # Test basic calculation
        result = calculator.calculate_s_value(2.0)
        print(f"✓ S-entropy calculation: S = k * log(2.0) = {result:.3f}")
        
        # Test coordinate conversion
        coords = calculator.amplitude_to_coordinates(1.5)
        print(f"✓ Coordinate conversion: amplitude 1.5 → K={coords.knowledge:.3f}, T={coords.time:.3f}, E={coords.entropy:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ S-entropy functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_environmental_measurement():
    """Test environmental measurement system."""
    print("\nTesting environmental measurement...")
    
    try:
        from graffiti.environmental.twelve_dimensional import EnvironmentalMeasurement
        
        async def run_env_test():
            measurement_system = EnvironmentalMeasurement()
            env_state = await measurement_system.measure_environment()
            print(f"✓ Environmental measurement: uniqueness = {env_state.calculate_uniqueness():.6f}")
            return True
        
        return asyncio.run(run_env_test())
    except Exception as e:
        print(f"✗ Environmental measurement test failed: {e}")
        traceback.print_exc()
        return False


def test_meaning_impossibility():
    """Test meaning impossibility analyzer."""
    print("\nTesting meaning impossibility analysis...")
    
    try:
        from graffiti.core.meaning_impossibility import MeaningImpossibilityAnalyzer
        from graffiti.core.types import Query
        
        analyzer = MeaningImpossibilityAnalyzer()
        query = Query(content="What is the meaning of existence?")
        result = analyzer.analyze_meaning_requirements(query)
        
        print(f"✓ Meaning impossibility analysis: {result.score:.1%} impossibility demonstrated")
        return True
    except Exception as e:
        print(f"✗ Meaning impossibility test failed: {e}")
        traceback.print_exc()
        return False


async def test_search_demo():
    """Test basic search demo functionality."""
    print("\nTesting search demo...")
    
    try:
        from graffiti.applications.search_demo import GraffitiSearchDemo
        from graffiti.core.types import ProcessingMode
        
        demo = GraffitiSearchDemo()
        result = await demo.search("test query", ProcessingMode.S_ENTROPY_NAVIGATION)
        
        print(f"✓ Search demo: processed query in {result['performance']['total_processing_time']:.3f}s")
        print(f"  Frameworks used: {result['performance']['frameworks_processed']}")
        print(f"  Environmental uniqueness: {result['environmental_context']['uniqueness']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Search demo test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all installation tests."""
    print("GRAFFITI RESEARCH DEMO - INSTALLATION TEST")
    print("="*50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Basic imports
    if test_basic_imports():
        tests_passed += 1
    
    # Test 2: S-entropy functionality
    if test_s_entropy_basic_functionality():
        tests_passed += 1
    
    # Test 3: Environmental measurement
    if test_environmental_measurement():
        tests_passed += 1
    
    # Test 4: Meaning impossibility
    if test_meaning_impossibility():
        tests_passed += 1
    
    # Test 5: Search demo
    async def run_search_test():
        return await test_search_demo()
    
    if asyncio.run(run_search_test()):
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"INSTALLATION TEST RESULTS")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ ALL TESTS PASSED - Package installation successful!")
        print("\nYou can now run:")
        print("  python -m graffiti.applications.search_demo --demo")
        print("  graffiti-demo --query 'your query here'")
        print("  python demo_revolutionary_search.py")
        return 0
    else:
        print("✗ Some tests failed - Package may have installation issues")
        print("Try: pip install -e . --force-reinstall")
        return 1


if __name__ == "__main__":
    sys.exit(main())
