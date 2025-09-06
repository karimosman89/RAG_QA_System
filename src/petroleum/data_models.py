"""
Petroleum Data Models and Processing

Specialized data models for oil & gas operations including:
- Well log data structures and validation
- Drilling parameter monitoring
- Reservoir modeling and properties
- Geological formation classification
- Real-time data processing pipelines
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid

# Enums for petroleum industry standards
class LogType(str, Enum):
    """Standard well log types."""
    GAMMA_RAY = "gamma_ray"
    RESISTIVITY = "resistivity"
    NEUTRON = "neutron"
    DENSITY = "density"
    PHOTOELECTRIC = "photoelectric"
    SP = "spontaneous_potential"
    CALIPER = "caliper"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    MUD_LOG = "mud_log"
    IMAGE_LOG = "image_log"
    NMR = "nuclear_magnetic_resonance"
    ACOUSTIC = "acoustic"
    FORMATION_TESTER = "formation_tester"


class LithologyType(str, Enum):
    """Rock type classifications."""
    SANDSTONE = "sandstone"
    SHALE = "shale"
    LIMESTONE = "limestone"
    DOLOMITE = "dolomite"
    COAL = "coal"
    SALT = "salt"
    ANHYDRITE = "anhydrite"
    GRANITE = "granite"
    BASALT = "basalt"
    CONGLOMERATE = "conglomerate"
    UNKNOWN = "unknown"


class FluidType(str, Enum):
    """Fluid types in reservoirs."""
    OIL = "oil"
    GAS = "gas"
    WATER = "water"
    BRINE = "brine"
    CONDENSATE = "condensate"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DrillingHazard(str, Enum):
    """Drilling hazard types."""
    KICK = "kick"
    LOST_CIRCULATION = "lost_circulation"
    STUCK_PIPE = "stuck_pipe"
    WASHOUT = "washout"
    TIGHT_HOLE = "tight_hole"
    PACK_OFF = "pack_off"
    TWIST_OFF = "twist_off"
    H2S = "hydrogen_sulfide"
    CO2 = "carbon_dioxide"
    HIGH_PRESSURE = "high_pressure"
    UNSTABLE_FORMATION = "unstable_formation"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Data Models
class WellLogData(BaseModel):
    """Well log data structure with validation."""
    
    well_id: str = Field(..., description="Unique well identifier")
    log_type: LogType = Field(..., description="Type of log measurement")
    depth_start: float = Field(..., gt=0, description="Starting depth in feet/meters")
    depth_end: float = Field(..., gt=0, description="Ending depth in feet/meters")
    depth_unit: str = Field(default="ft", pattern="^(ft|m)$", description="Depth unit")
    
    # Log data arrays
    depths: List[float] = Field(..., min_items=1, description="Depth measurements")
    values: List[float] = Field(..., min_items=1, description="Log values")
    quality_flags: Optional[List[int]] = Field(None, description="Data quality flags")
    
    # Metadata
    acquisition_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    logging_company: Optional[str] = None
    tool_type: Optional[str] = None
    run_number: Optional[int] = None
    
    # Processing parameters
    sampling_rate: Optional[float] = Field(None, gt=0, description="Sampling rate in Hz")
    vertical_resolution: Optional[float] = Field(None, gt=0, description="Vertical resolution")
    
    # Calibration and corrections
    calibration_data: Dict[str, Any] = Field(default_factory=dict)
    environmental_corrections: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('values')
    def validate_values_length(cls, v, values):
        if 'depths' in values and len(v) != len(values['depths']):
            raise ValueError('Values and depths arrays must have same length')
        return v
    
    @validator('depth_end')
    def validate_depth_end(cls, v, values):
        if 'depth_start' in values and v <= values['depth_start']:
            raise ValueError('Depth end must be greater than depth start')
        return v
    
    def get_depth_range(self) -> Tuple[float, float]:
        """Get the depth range of the log."""
        return (min(self.depths), max(self.depths))
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics of log values."""
        values_array = np.array(self.values)
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "p10": float(np.percentile(values_array, 10)),
            "p90": float(np.percentile(values_array, 90))
        }


class DrillingParameters(BaseModel):
    """Real-time drilling parameters monitoring."""
    
    well_id: str = Field(..., description="Well identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    measured_depth: float = Field(..., ge=0, description="Current measured depth")
    
    # Drilling mechanics
    weight_on_bit: float = Field(..., ge=0, description="Weight on bit (klbs)")
    rotary_speed: float = Field(..., ge=0, description="Rotary speed (RPM)")
    rate_of_penetration: float = Field(..., ge=0, description="ROP (ft/hr)")
    torque: float = Field(..., ge=0, description="Torque (ft-lbs)")
    
    # Hydraulics
    pump_pressure: float = Field(..., ge=0, description="Pump pressure (psi)")
    flow_rate: float = Field(..., ge=0, description="Flow rate (gpm)")
    
    # Mud properties
    mud_weight: float = Field(..., gt=0, description="Mud weight (ppg)")
    mud_viscosity: float = Field(..., gt=0, description="Mud viscosity (sec)")
    mud_temperature: float = Field(..., description="Mud temperature (°F)")
    
    # Formation evaluation
    gamma_ray: Optional[float] = Field(None, description="Real-time gamma ray")
    resistivity: Optional[float] = Field(None, description="Real-time resistivity")
    
    # Gas monitoring
    total_gas: Optional[float] = Field(None, ge=0, description="Total gas (units)")
    c1_gas: Optional[float] = Field(None, ge=0, description="C1 gas (ppm)")
    c2_gas: Optional[float] = Field(None, ge=0, description="C2 gas (ppm)")
    c3_gas: Optional[float] = Field(None, ge=0, description="C3 gas (ppm)")
    
    # Mechanical specific energy
    mse: Optional[float] = Field(None, description="Mechanical Specific Energy")
    
    # Alerts and warnings
    active_alerts: List[str] = Field(default_factory=list)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    
    def calculate_mse(self) -> float:
        """Calculate Mechanical Specific Energy."""
        if self.rate_of_penetration > 0:
            # Simplified MSE calculation
            mse = (self.weight_on_bit * 1000 + (4 * np.pi * self.torque * self.rotary_speed) / 
                   (12 * self.rate_of_penetration)) / (np.pi * (8.5 ** 2) / 4)
            return round(mse, 2)
        return 0.0
    
    def detect_drilling_dysfunction(self) -> List[str]:
        """Detect potential drilling dysfunctions."""
        dysfunctions = []
        
        # Calculate MSE if not provided
        mse = self.mse or self.calculate_mse()
        
        # High MSE indicates inefficient drilling
        if mse > 300000:  # psi threshold
            dysfunctions.append("High MSE - Inefficient drilling")
        
        # Stick-slip detection (high torque variation)
        if self.torque > 30000:  # ft-lbs threshold
            dysfunctions.append("Potential stick-slip")
        
        # Low ROP with high WOB
        if self.rate_of_penetration < 10 and self.weight_on_bit > 30:
            dysfunctions.append("Low ROP with high WOB")
        
        return dysfunctions


class ReservoirModel(BaseModel):
    """Reservoir properties and modeling data."""
    
    reservoir_id: str = Field(..., description="Reservoir identifier")
    well_id: str = Field(..., description="Associated well")
    
    # Structural properties
    top_depth: float = Field(..., description="Top of reservoir (ft)")
    bottom_depth: float = Field(..., description="Bottom of reservoir (ft)")
    net_thickness: float = Field(..., gt=0, description="Net pay thickness (ft)")
    
    # Petrophysical properties
    porosity: float = Field(..., ge=0, le=1, description="Average porosity (fraction)")
    permeability: float = Field(..., gt=0, description="Average permeability (mD)")
    water_saturation: float = Field(..., ge=0, le=1, description="Water saturation")
    
    # Fluid properties
    primary_fluid: FluidType = Field(..., description="Primary fluid type")
    oil_api_gravity: Optional[float] = Field(None, description="Oil API gravity")
    gas_gravity: Optional[float] = Field(None, description="Gas specific gravity")
    
    # Pressure and temperature
    reservoir_pressure: Optional[float] = Field(None, description="Reservoir pressure (psi)")
    reservoir_temperature: Optional[float] = Field(None, description="Temperature (°F)")
    
    # Geological properties
    lithology: LithologyType = Field(..., description="Primary lithology")
    depositional_environment: Optional[str] = None
    
    # Production potential
    estimated_reserves: Optional[float] = Field(None, description="Estimated reserves")
    productivity_index: Optional[float] = Field(None, description="Productivity index")
    
    # Uncertainty and confidence
    confidence_level: float = Field(default=0.5, ge=0, le=1, description="Model confidence")
    uncertainty_range: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    
    def calculate_hydrocarbon_saturation(self) -> float:
        """Calculate hydrocarbon saturation."""
        return 1.0 - self.water_saturation
    
    def estimate_original_oil_in_place(self, area_acres: float) -> Optional[float]:
        """Estimate original oil in place (MMSTB)."""
        if self.primary_fluid in [FluidType.OIL, FluidType.CONDENSATE]:
            # OOIP = 7758 × Area × Net_thickness × Porosity × (1 - Sw) / Bo
            # Simplified calculation assuming Bo = 1.2
            bo = 1.2  # Formation volume factor
            ooip = (7758 * area_acres * self.net_thickness * self.porosity * 
                   self.calculate_hydrocarbon_saturation()) / bo / 1000000  # Convert to MMSTB
            return round(ooip, 2)
        return None


class GeologicalFormation(BaseModel):
    """Geological formation properties and interpretation."""
    
    formation_id: str = Field(..., description="Formation identifier")
    formation_name: str = Field(..., description="Formation name")
    well_id: str = Field(..., description="Associated well")
    
    # Depth interval
    top_depth: float = Field(..., description="Formation top (ft)")
    bottom_depth: float = Field(..., description="Formation bottom (ft)")
    thickness: float = Field(..., gt=0, description="Formation thickness (ft)")
    
    # Lithology and properties
    primary_lithology: LithologyType = Field(..., description="Primary rock type")
    secondary_lithology: Optional[LithologyType] = None
    lithology_percentages: Dict[str, float] = Field(default_factory=dict)
    
    # Depositional environment
    depositional_environment: Optional[str] = None
    age: Optional[str] = Field(None, description="Geological age")
    
    # Reservoir quality
    reservoir_quality: float = Field(default=0.0, ge=0, le=1, description="Reservoir quality index")
    seal_quality: float = Field(default=0.0, ge=0, le=1, description="Seal quality index")
    
    # Drilling considerations
    drilling_hazards: List[DrillingHazard] = Field(default_factory=list)
    recommended_mud_weight: Optional[float] = Field(None, description="Recommended mud weight (ppg)")
    
    # Log characteristics
    typical_gr_range: Optional[Tuple[float, float]] = None
    typical_resistivity_range: Optional[Tuple[float, float]] = None
    
    # Structural features
    dip_angle: Optional[float] = Field(None, ge=0, le=90, description="Formation dip (degrees)")
    strike_direction: Optional[float] = Field(None, ge=0, lt=360, description="Strike direction")
    
    # Interpretation confidence
    interpretation_confidence: float = Field(default=0.5, ge=0, le=1)
    interpreted_by: Optional[str] = None
    interpretation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_formation_thickness(self) -> float:
        """Calculate formation thickness."""
        return self.bottom_depth - self.top_depth
    
    def is_potential_reservoir(self) -> bool:
        """Determine if formation has reservoir potential."""
        return (self.reservoir_quality > 0.3 and 
                self.primary_lithology in [LithologyType.SANDSTONE, LithologyType.LIMESTONE, LithologyType.DOLOMITE])
    
    def get_drilling_risk_assessment(self) -> Dict[str, Any]:
        """Assess drilling risks for this formation."""
        risk_factors = []
        overall_risk = RiskLevel.LOW
        
        if DrillingHazard.HIGH_PRESSURE in self.drilling_hazards:
            risk_factors.append("High pressure zone")
            overall_risk = RiskLevel.HIGH
        
        if DrillingHazard.UNSTABLE_FORMATION in self.drilling_hazards:
            risk_factors.append("Unstable formation")
            if overall_risk == RiskLevel.LOW:
                overall_risk = RiskLevel.MEDIUM
        
        if self.primary_lithology == LithologyType.SHALE and self.thickness > 500:
            risk_factors.append("Thick shale section - potential instability")
            if overall_risk == RiskLevel.LOW:
                overall_risk = RiskLevel.MEDIUM
        
        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "recommended_mud_weight": self.recommended_mud_weight or 9.5,
            "special_considerations": self.drilling_hazards
        }


class PetroleumDataProcessor:
    """Advanced petroleum data processing and analysis."""
    
    def __init__(self):
        self.processed_wells = {}
        self.formation_database = {}
        self.correlation_models = {}
    
    def process_well_logs(self, well_logs: List[WellLogData]) -> Dict[str, Any]:
        """Process multiple well logs and generate integrated analysis."""
        
        if not well_logs:
            return {"error": "No well logs provided"}
        
        # Group logs by well_id
        wells_data = {}
        for log in well_logs:
            if log.well_id not in wells_data:
                wells_data[log.well_id] = {}
            wells_data[log.well_id][log.log_type] = log
        
        analysis_results = {}
        
        for well_id, logs in wells_data.items():
            well_analysis = {
                "well_id": well_id,
                "available_logs": list(logs.keys()),
                "depth_range": self._calculate_depth_range(logs),
                "log_statistics": {},
                "lithology_predictions": [],
                "fluid_contacts": [],
                "reservoir_zones": []
            }
            
            # Calculate statistics for each log type
            for log_type, log_data in logs.items():
                well_analysis["log_statistics"][log_type] = log_data.get_statistics()
            
            # Perform integrated analysis
            if LogType.GAMMA_RAY in logs and LogType.RESISTIVITY in logs:
                well_analysis["lithology_predictions"] = self._predict_lithology(
                    logs[LogType.GAMMA_RAY], logs[LogType.RESISTIVITY]
                )
            
            if LogType.RESISTIVITY in logs and LogType.NEUTRON in logs:
                well_analysis["fluid_contacts"] = self._detect_fluid_contacts(
                    logs[LogType.RESISTIVITY], logs.get(LogType.NEUTRON)
                )
            
            analysis_results[well_id] = well_analysis
        
        return analysis_results
    
    def _calculate_depth_range(self, logs: Dict[LogType, WellLogData]) -> Tuple[float, float]:
        """Calculate overall depth range for all logs."""
        min_depth = float('inf')
        max_depth = float('-inf')
        
        for log in logs.values():
            log_min, log_max = log.get_depth_range()
            min_depth = min(min_depth, log_min)
            max_depth = max(max_depth, log_max)
        
        return (min_depth, max_depth)
    
    def _predict_lithology(self, gr_log: WellLogData, res_log: WellLogData) -> List[Dict[str, Any]]:
        """Predict lithology using gamma ray and resistivity logs."""
        predictions = []
        
        # Simplified lithology prediction based on log characteristics
        gr_values = np.array(gr_log.values)
        res_values = np.array(res_log.values)
        depths = np.array(gr_log.depths)
        
        for i in range(len(gr_values)):
            prediction = {
                "depth": depths[i],
                "gamma_ray": gr_values[i],
                "resistivity": res_values[i] if i < len(res_values) else None,
                "predicted_lithology": LithologyType.UNKNOWN,
                "confidence": 0.0
            }
            
            # Simple rules-based lithology prediction
            if gr_values[i] < 50:  # Low gamma ray
                if i < len(res_values) and res_values[i] > 10:
                    prediction["predicted_lithology"] = LithologyType.SANDSTONE
                    prediction["confidence"] = 0.7
                else:
                    prediction["predicted_lithology"] = LithologyType.LIMESTONE
                    prediction["confidence"] = 0.6
            elif gr_values[i] > 100:  # High gamma ray
                prediction["predicted_lithology"] = LithologyType.SHALE
                prediction["confidence"] = 0.8
            else:  # Medium gamma ray
                prediction["predicted_lithology"] = LithologyType.SANDSTONE
                prediction["confidence"] = 0.5
            
            predictions.append(prediction)
        
        return predictions
    
    def _detect_fluid_contacts(self, res_log: WellLogData, neutron_log: Optional[WellLogData] = None) -> List[Dict[str, Any]]:
        """Detect potential fluid contacts using resistivity and neutron logs."""
        contacts = []
        
        res_values = np.array(res_log.values)
        depths = np.array(res_log.depths)
        
        # Look for significant resistivity changes (potential fluid contacts)
        for i in range(1, len(res_values) - 1):
            # Calculate resistivity gradient
            gradient = abs(res_values[i+1] - res_values[i-1]) / (depths[i+1] - depths[i-1])
            
            # High gradient indicates potential contact
            if gradient > np.percentile(np.abs(np.gradient(res_values)), 95):
                contact_type = FluidType.UNKNOWN
                confidence = 0.5
                
                # Determine contact type based on resistivity behavior
                if res_values[i+1] > res_values[i-1] * 2:
                    contact_type = FluidType.OIL  # Resistivity increases downward
                    confidence = 0.7
                elif res_values[i+1] < res_values[i-1] / 2:
                    contact_type = FluidType.WATER  # Resistivity decreases downward
                    confidence = 0.7
                
                contacts.append({
                    "depth": depths[i],
                    "contact_type": contact_type,
                    "resistivity_above": res_values[i-1],
                    "resistivity_below": res_values[i+1],
                    "confidence": confidence
                })
        
        return contacts
    
    def correlate_wells(self, formations: List[GeologicalFormation]) -> Dict[str, Any]:
        """Correlate formations between wells."""
        correlation_results = {
            "correlated_formations": {},
            "structural_trends": [],
            "depositional_environments": {}
        }
        
        # Group formations by name
        formation_groups = {}
        for formation in formations:
            name = formation.formation_name
            if name not in formation_groups:
                formation_groups[name] = []
            formation_groups[name].append(formation)
        
        # Analyze each formation group
        for formation_name, group_formations in formation_groups.items():
            if len(group_formations) >= 2:  # Need at least 2 wells for correlation
                correlation_data = {
                    "formation_name": formation_name,
                    "wells_count": len(group_formations),
                    "depth_variations": [],
                    "thickness_variations": [],
                    "structural_dip": None
                }
                
                # Calculate depth and thickness variations
                depths = [f.top_depth for f in group_formations]
                thicknesses = [f.thickness for f in group_formations]
                
                correlation_data["depth_variations"] = {
                    "min_depth": min(depths),
                    "max_depth": max(depths),
                    "depth_range": max(depths) - min(depths)
                }
                
                correlation_data["thickness_variations"] = {
                    "min_thickness": min(thicknesses),
                    "max_thickness": max(thicknesses),
                    "avg_thickness": np.mean(thicknesses),
                    "thickness_std": np.std(thicknesses)
                }
                
                correlation_results["correlated_formations"][formation_name] = correlation_data
        
        return correlation_results


# Global data processor instance
petroleum_data_processor = PetroleumDataProcessor()