"""
Appliance Configuration

Default parameters and thresholds for common appliances.
Based on REDD, UK-DALE, and other NILM datasets.
"""

from typing import Dict, Any

# Power thresholds and parameters for common appliances
APPLIANCE_PARAMS: Dict[str, Dict[str, Any]] = {
    # Kitchen appliances
    "refrigerator": {
        "min_power": 0,
        "max_power": 400,
        "on_threshold": 50,
        "mean_power": 120,
        "standby_power": 5,
        "typical_duration": "continuous",
        "category": "always_on",
    },
    "microwave": {
        "min_power": 0,
        "max_power": 2000,
        "on_threshold": 200,
        "mean_power": 1000,
        "standby_power": 3,
        "typical_duration": "1-5 minutes",
        "category": "short_cycle",
    },
    "dishwasher": {
        "min_power": 0,
        "max_power": 2500,
        "on_threshold": 10,
        "mean_power": 700,
        "standby_power": 2,
        "typical_duration": "60-90 minutes",
        "category": "long_cycle",
    },
    "kettle": {
        "min_power": 0,
        "max_power": 3100,
        "on_threshold": 2000,
        "mean_power": 2800,
        "standby_power": 0,
        "typical_duration": "2-4 minutes",
        "category": "short_cycle",
    },
    "toaster": {
        "min_power": 0,
        "max_power": 1500,
        "on_threshold": 100,
        "mean_power": 900,
        "standby_power": 0,
        "typical_duration": "1-3 minutes",
        "category": "short_cycle",
    },
    "oven": {
        "min_power": 0,
        "max_power": 4000,
        "on_threshold": 100,
        "mean_power": 2000,
        "standby_power": 5,
        "typical_duration": "20-60 minutes",
        "category": "variable",
    },
    
    # Laundry appliances
    "washing_machine": {
        "min_power": 0,
        "max_power": 2500,
        "on_threshold": 20,
        "mean_power": 400,
        "standby_power": 2,
        "typical_duration": "30-90 minutes",
        "category": "long_cycle",
    },
    "dryer": {
        "min_power": 0,
        "max_power": 3500,
        "on_threshold": 100,
        "mean_power": 2500,
        "standby_power": 2,
        "typical_duration": "40-60 minutes",
        "category": "long_cycle",
    },
    
    # Climate control
    "air_conditioner": {
        "min_power": 0,
        "max_power": 3500,
        "on_threshold": 100,
        "mean_power": 1500,
        "standby_power": 5,
        "typical_duration": "variable",
        "category": "variable",
    },
    "heater": {
        "min_power": 0,
        "max_power": 3000,
        "on_threshold": 100,
        "mean_power": 2000,
        "standby_power": 0,
        "typical_duration": "variable",
        "category": "variable",
    },
    "fan": {
        "min_power": 0,
        "max_power": 200,
        "on_threshold": 10,
        "mean_power": 50,
        "standby_power": 0,
        "typical_duration": "variable",
        "category": "variable",
    },
    
    # Electronics
    "television": {
        "min_power": 0,
        "max_power": 200,
        "on_threshold": 20,
        "mean_power": 100,
        "standby_power": 5,
        "typical_duration": "1-4 hours",
        "category": "entertainment",
    },
    "computer": {
        "min_power": 0,
        "max_power": 400,
        "on_threshold": 30,
        "mean_power": 150,
        "standby_power": 5,
        "typical_duration": "variable",
        "category": "variable",
    },
    "laptop": {
        "min_power": 0,
        "max_power": 100,
        "on_threshold": 15,
        "mean_power": 50,
        "standby_power": 2,
        "typical_duration": "variable",
        "category": "variable",
    },
    "game_console": {
        "min_power": 0,
        "max_power": 200,
        "on_threshold": 30,
        "mean_power": 120,
        "standby_power": 5,
        "typical_duration": "1-3 hours",
        "category": "entertainment",
    },
    
    # Lighting
    "light": {
        "min_power": 0,
        "max_power": 200,
        "on_threshold": 5,
        "mean_power": 40,
        "standby_power": 0,
        "typical_duration": "variable",
        "category": "variable",
    },
    
    # Other
    "electric_vehicle_charger": {
        "min_power": 0,
        "max_power": 11000,
        "on_threshold": 100,
        "mean_power": 7000,
        "standby_power": 5,
        "typical_duration": "4-8 hours",
        "category": "long_cycle",
    },
    "water_heater": {
        "min_power": 0,
        "max_power": 4500,
        "on_threshold": 100,
        "mean_power": 4000,
        "standby_power": 0,
        "typical_duration": "15-45 minutes",
        "category": "long_cycle",
    },
}


def get_appliance_params(appliance_name: str) -> Dict[str, Any]:
    """
    Get parameters for a specific appliance.
    
    Args:
        appliance_name: Name of the appliance
        
    Returns:
        Dictionary of appliance parameters
    """
    name_lower = appliance_name.lower().replace(" ", "_")
    
    if name_lower in APPLIANCE_PARAMS:
        return APPLIANCE_PARAMS[name_lower].copy()
    
    # Try fuzzy matching
    for key in APPLIANCE_PARAMS:
        if key in name_lower or name_lower in key:
            return APPLIANCE_PARAMS[key].copy()
    
    # Return default parameters
    return {
        "min_power": 0,
        "max_power": 3000,
        "on_threshold": 10,
        "mean_power": 500,
        "standby_power": 0,
        "typical_duration": "unknown",
        "category": "unknown",
    }


def get_on_threshold(appliance_name: str) -> float:
    """Get the on/off threshold for an appliance."""
    params = get_appliance_params(appliance_name)
    return params.get("on_threshold", 10.0)


def get_max_power(appliance_name: str) -> float:
    """Get the maximum expected power for an appliance."""
    params = get_appliance_params(appliance_name)
    return params.get("max_power", 3000.0)


# Appliance sets for common datasets
REDD_APPLIANCES = [
    "refrigerator",
    "dishwasher", 
    "microwave",
    "light",
    "washer_dryer",
]

UKDALE_APPLIANCES = [
    "kettle",
    "microwave",
    "fridge",
    "dishwasher",
    "washing_machine",
]

REFIT_APPLIANCES = [
    "fridge_freezer",
    "washing_machine",
    "dishwasher",
    "television",
    "microwave",
    "kettle",
    "toaster",
    "boiler",
    "computer",
]
