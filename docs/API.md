# API Documentation

## Core Functions

### unified_map_cell.py

#### `calculate_bacteria_comfort_index(conditions, disease_profile)`

Calculates the Bacteria Comfort Index - a proprietary metric assessing environmental suitability for pathogen survival and reproduction.

**Parameters:**
- `conditions` (dict): Zone satellite and climate data
  - `satellite_data` (dict): Contains temperature, humidity, NDVI, NDWI, precipitation
- `disease_profile` (dict): Disease-specific environmental requirements
  - `environmental_factors` (dict): Optimal ranges for temperature, humidity, etc.

**Returns:**
- `dict`: Comfort index breakdown
  - `overall_comfort` (float): 0-100% overall suitability
  - `comfort_level` (str): VERY LOW, LOW, MODERATE, HIGH, VERY HIGH
  - `temperature_comfort` (float): Temperature factor (0-100%)
  - `humidity_comfort` (float): Humidity factor (0-100%)
  - `water_comfort` (float): Water availability factor (0-100%)
  - `vegetation_comfort` (float): Vegetation factor (0-100%)

**Example:**
```python
comfort = calculate_bacteria_comfort_index(zone_data, disease_profile)
print(f"Risk Level: {comfort['comfort_level']}")
print(f"Overall: {comfort['overall_comfort']:.1f}%")
```

---

#### `create_unified_map(vaccination_plan, disease, output_path='unified_satellite_lives_map.html')`

Creates an interactive 4-layer Folium map with risk analysis and vaccination planning.

**Parameters:**
- `vaccination_plan` (list): List of zone dictionaries with coordinates, population, satellite data
- `disease` (dict): Disease profile with parameters
- `output_path` (str, optional): Output HTML file path

**Returns:**
- `folium.Map`: Interactive map object

**Outputs:**
- `{output_path}` - Interactive HTML map
- `unified_analysis_charts.png` - 7 analytical charts
- `unified_satellite_comfort_lives_data.csv` - Detailed zone data (23 columns)

**Map Layers:**
1. Lives Saved & Priorities - Circles sized by lives saved, colored by priority
2. Bacteria Comfort Heatmap - Risk gradient (green→red)
3. Lives Saved Heatmap - Impact visualization
4. Satellite Data Points - Detailed environmental data markers

**Example:**
```python
map_obj = create_unified_map(
    vaccination_plan=zones,
    disease=disease_profile,
    output_path='my_map.html'
)
```

---

### ai_components.py

#### `integrate_ai_components(vaccination_plan, disease, summary)`

Adds ML predictions and LLM analysis to the epidemic prediction system.

**Parameters:**
- `vaccination_plan` (list): Zone data from SEIR model
- `disease` (dict): Disease parameters
- `summary` (dict): Overall statistics

**Returns:**
- `tuple`: (ai_results, enhanced_vaccination_plan)
  - `ai_results` (dict): Contains ML predictor, AI assistant, feature importances
  - `enhanced_vaccination_plan` (list): Zones with ML predictions added

**Example:**
```python
ai_results, enhanced_plan = integrate_ai_components(
    vaccination_plan, disease_profile, summary
)

# Use ML predictor
predictor = ai_results['ml_predictor']

# Use AI assistant
assistant = ai_results['ai_assistant']
answer = assistant.ask("Which zones are critical?")
```

---

#### `class OutbreakPredictor`

Random Forest ML model for predicting outbreak probability.

**Methods:**

##### `create_synthetic_training_data(vaccination_plan, n_samples=500)`
Creates synthetic training data for ML model.

**Parameters:**
- `vaccination_plan` (list): Current zone data
- `n_samples` (int): Number of training samples to generate

**Returns:**
- `pd.DataFrame`: Training dataset with features and labels

---

##### `train(training_data)`
Trains the Random Forest model.

**Parameters:**
- `training_data` (pd.DataFrame): Dataset from create_synthetic_training_data()

**Returns:**
- `pd.DataFrame`: Feature importances

**Prints:**
- Classification report
- ROC-AUC score
- Confusion matrix
- Top feature importances

---

##### `predict(zone_data)`
Predicts outbreak probability for a zone.

**Parameters:**
- `zone_data` (dict): Zone with satellite_data and bacteria_comfort

**Returns:**
- `float`: Outbreak probability (0-100%)

**Example:**
```python
predictor = OutbreakPredictor()
training_data = predictor.create_synthetic_training_data(vaccination_plan)
predictor.train(training_data)

probability = predictor.predict(zone_data)
print(f"Outbreak probability: {probability:.1f}%")
```

---

#### `class AIEpidemicAssistant`

LLM-powered assistant using Anthropic Claude for epidemic analysis.

**Methods:**

##### `__init__(vaccination_plan, disease_profile, summary)`
Initializes the AI assistant.

**Parameters:**
- `vaccination_plan` (list): Zone data
- `disease_profile` (dict): Disease parameters
- `summary` (dict): Overall statistics

---

##### `ask(question)`
Ask the AI assistant a question.

**Parameters:**
- `question` (str): Question in natural language (English or Russian)

**Returns:**
- `str`: AI-generated answer

**Example:**
```python
assistant = AIEpidemicAssistant(vaccination_plan, disease, summary)
answer = assistant.ask("What should we do first?")
print(answer)
```

---

##### `generate_executive_summary()`
Generates a 1-page executive summary for government officials.

**Returns:**
- `str`: Executive summary text

**Example:**
```python
summary = assistant.generate_executive_summary()
print(summary)
```

---

##### `explain_ml_prediction(zone_number)`
Explains why ML predicted high/low risk for a specific zone.

**Parameters:**
- `zone_number` (int): Zone ID

**Returns:**
- `str`: Natural language explanation

---

## Data Structures

### Zone Dictionary

```python
zone = {
    'zone_number': 1,
    'coordinates': (14.5995, 120.9842),  # (lat, lon)
    'population': 50000,
    'priority': 'CRITICAL',  # or HIGH, MEDIUM, LOW
    
    'satellite_data': {
        'temperature': 28.5,      # Celsius
        'humidity': 85,           # Percentage
        'ndvi': 0.45,            # -1 to 1
        'ndwi': 0.25,            # -1 to 1
        'precipitation_30d': 150, # mm
        'water_present': True,
        'data_source': 'Sentinel-2'
    },
    
    'bacteria_comfort': {
        'overall_comfort': 87.5,         # 0-100%
        'comfort_level': 'VERY HIGH',
        'temperature_comfort': 95.0,
        'humidity_comfort': 92.0,
        'water_comfort': 100.0,
        'vegetation_comfort': 85.0
    },
    
    'vaccination': {
        'target_population': 35000,
        'primary_doses': 70000,
        'booster_doses': 35000,
        'total_doses': 115500,
        'coverage': 0.70
    },
    
    'impact': {
        'lives_saved': 245,
        'cases_prevented': 4900,
        'total_cost_usd': 866250,
        'cost_per_life': 3540
    },
    
    # Optional: Added by ML component
    'ml_outbreak_probability': 89.4,  # 0-100%
    'ml_risk': 'CRITICAL'
}
```

### Disease Profile Dictionary

```python
disease_profile = {
    'name': 'Leptospirosis',
    'pathogen': 'Leptospira bacteria',
    
    'R0': 1.8,                    # Basic reproduction number
    'fatality_rate': 0.05,        # 5%
    'incubation_period': 10,      # days
    'infectious_period': 7,       # days
    
    'vaccine_available': True,
    'vaccine_efficacy': 0.70,     # 70%
    
    'environmental_factors': {
        'temperature': {
            'optimal': 28,        # Celsius
            'range': (25, 35)
        },
        'humidity': {
            'optimal': 82.5,      # Percentage
            'range': (70, 95)
        },
        'ndvi_preference': 0.5,   # Preferred vegetation
        'water_dependent': True
    }
}
```

### Summary Dictionary

```python
summary = {
    'total_zones': 45,
    'total_population': 1250000,
    'cases_without_vaccine': 75432,
    'cases_with_vaccine': 28651,
    'prevented_cases': 46781,
    'prevented_deaths': 2339,
    'total_cost_usd': 19687500,
    'cost_per_prevented_death': 8417
}
```

## CSV Export Columns

The `unified_satellite_comfort_lives_data.csv` contains 23 columns:

1. `Zone` - Zone number
2. `Priority` - CRITICAL, HIGH, MEDIUM, LOW
3. `Lat`, `Lon` - Coordinates
4. `Population` - Total population
5. `Lives_Saved` - Expected lives saved
6. `Prevented_Cases` - Cases prevented by vaccination
7. `Cost_USD` - Total vaccination cost
8. `Cost_Per_Life` - Cost per life saved
9. `Comfort_Overall` - Overall Bacteria Comfort Index (%)
10. `Comfort_Level` - Risk category
11. `Comfort_Temperature` - Temperature factor (%)
12. `Comfort_Humidity` - Humidity factor (%)
13. `Comfort_Water` - Water factor (%)
14. `Comfort_Vegetation` - Vegetation factor (%)
15. `Temperature_C` - Actual temperature
16. `Humidity_Percent` - Actual humidity
17. `NDVI` - Normalized Difference Vegetation Index
18. `NDWI` - Normalized Difference Water Index
19. `Water_Present` - Boolean
20. `Precipitation_30d_mm` - 30-day precipitation
21. `Data_Source` - Sentinel-2 or Landsat
22. `ML_Probability` - ML outbreak probability (if AI enabled)
23. `ML_Risk` - ML risk category (if AI enabled)

## Error Handling

All functions include error handling. Common exceptions:

- `ValueError`: Invalid input parameters
- `KeyError`: Missing required data fields
- `RuntimeError`: API failures (Earth Engine, Claude)

Example:
```python
try:
    map_obj = create_unified_map(vaccination_plan, disease)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## Performance Notes

- **Map generation**: 10-30 seconds depending on number of zones
- **ML training**: 5-10 seconds for 500 samples
- **AI queries**: 2-5 seconds per question
- **Satellite data fetching**: 5-15 seconds per zone (from Earth Engine)

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `earthengine-api` - Google Earth Engine
- `folium` - Interactive maps
- `scikit-learn` - Machine learning
- `anthropic` - Claude AI
- `pandas`, `numpy` - Data processing
