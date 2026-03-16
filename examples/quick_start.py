"""
Quick Start Example for AI Epidemic Prediction System

This example shows how to use the system to create an interactive epidemic risk map.
"""

from src.unified_map_cell import (
    calculate_bacteria_comfort_index,
    create_unified_map,
    add_ml_predictions_to_zones
)
from src.ai_components import integrate_ai_components

# ============================================================================
# EXAMPLE 1: Basic Visualization
# ============================================================================

def basic_example(vaccination_plan, disease_profile):
    """
    Create a basic interactive map with Bacteria Comfort Index
    
    Args:
        vaccination_plan: List of zones with satellite data
        disease_profile: Disease parameters
    """
    
    # Create unified map
    map_html = create_unified_map(
        vaccination_plan=vaccination_plan,
        disease=disease_profile,
        output_path='epidemic_risk_map.html'
    )
    
    print("✅ Map created: epidemic_risk_map.html")
    print("📊 Charts created: unified_analysis_charts.png")
    print("📋 Data exported: unified_satellite_comfort_lives_data.csv")
    
    return map_html


# ============================================================================
# EXAMPLE 2: With AI Components
# ============================================================================

def ai_enhanced_example(vaccination_plan, disease_profile, summary):
    """
    Add ML predictions and AI assistant to the system
    
    Args:
        vaccination_plan: List of zones with satellite data
        disease_profile: Disease parameters
        summary: Summary statistics
    """
    
    # Integrate AI components
    ai_results, enhanced_plan = integrate_ai_components(
        vaccination_plan=vaccination_plan,
        disease=disease_profile,
        summary=summary
    )
    
    # Use AI assistant
    assistant = ai_results['ai_assistant']
    
    # Ask questions
    print("\n AI Assistant Demo:")
    
    questions = [
        "Which zones need immediate vaccination?",
        "What is the total expected cost?",
        "How many lives can we save?"
    ]
    
    for q in questions:
        print(f"\n❓ {q}")
        answer = assistant.ask(q)
        print(f"💬 {answer[:200]}...")  # First 200 chars
    
    # Get ML predictions
    ml_predictor = ai_results['ml_predictor']
    print(f"\n ML model trained with {len(enhanced_plan)} zones")
    
    # Create enhanced map
    create_unified_map(
        vaccination_plan=enhanced_plan,
        disease=disease_profile,
        output_path='ai_enhanced_map.html'
    )
    
    return ai_results, enhanced_plan


# ============================================================================
# EXAMPLE 3: Calculate Bacteria Comfort Index for a Single Zone
# ============================================================================

def single_zone_example():
    """
    Calculate Bacteria Comfort Index for a single zone
    """
    
    # Example zone data
    zone_data = {
        'satellite_data': {
            'temperature': 28.5,
            'humidity': 85,
            'ndvi': 0.45,
            'ndwi': 0.25,
            'precipitation_30d': 150
        }
    }
    
    # Example disease profile (Leptospirosis)
    disease_profile = {
        'name': 'Leptospirosis',
        'environmental_factors': {
            'temperature': {
                'optimal': 28,
                'range': (25, 35)
            },
            'humidity': {
                'optimal': 82.5,
                'range': (70, 95)
            },
            'ndvi_preference': 0.5,
            'water_dependent': True
        }
    }
    
    # Calculate comfort index
    comfort = calculate_bacteria_comfort_index(
        zone_data,
        disease_profile
    )
    
    print("\n Bacteria Comfort Index Analysis:")
    print(f"Overall Comfort: {comfort['overall_comfort']:.1f}%")
    print(f"Risk Level: {comfort['comfort_level']}")
    print(f"\nFactors:")
    print(f"  Temperature: {comfort['temperature_comfort']:.1f}%")
    print(f"  Humidity: {comfort['humidity_comfort']:.1f}%")
    print(f"  Water: {comfort['water_comfort']:.1f}%")
    print(f"  Vegetation: {comfort['vegetation_comfort']:.1f}%")
    
    return comfort


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("AI EPIDEMIC PREDICTION SYSTEM - QUICK START EXAMPLES")
    print("="*70)
    
    # Example 3: Single zone calculation
    print("\n📍 Example: Single Zone Analysis")
    single_zone_example()
    
    print("\n" + "="*70)
    print("For full examples, you need:")
    print("1. vaccination_plan from your SEIR model")
    print("2. disease_profile with parameters")
    print("3. summary statistics")
    print("="*70)
