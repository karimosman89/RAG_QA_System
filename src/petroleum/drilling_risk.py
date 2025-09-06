"""
Drilling Risk Prediction System with Real-Time Monitoring

Advanced AI system for predicting and monitoring drilling hazards:
- Real-time risk assessment using ML models and physics-based analysis
- Predictive analytics for kick detection, lost circulation, and stuck pipe
- Geomechanical modeling for wellbore stability analysis
- Real-time drilling parameter optimization recommendations
- Automated alert system with severity classification
- Historical drilling data analysis and trend identification
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from collections import deque

# ML and statistical libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from scipy import stats
from scipy.signal import find_peaks
import joblib

# Streaming data processing
import asyncio
from asyncio import Queue

from .data_models import (
    DrillingParameters, DrillingHazard, RiskLevel, 
    LithologyType, GeologicalFormation
)

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskCategory(str, Enum):
    """Categories of drilling risks."""
    MECHANICAL = "mechanical"
    HYDRAULIC = "hydraulic"
    FORMATION = "formation"
    WELLBORE_STABILITY = "wellbore_stability"
    FLUID_MANAGEMENT = "fluid_management"
    EQUIPMENT = "equipment"


@dataclass
class RiskAlert:
    """Drilling risk alert structure."""
    alert_id: str
    timestamp: datetime
    risk_type: DrillingHazard
    severity: AlertSeverity
    category: RiskCategory
    probability: float
    confidence: float
    current_depth: float
    description: str
    recommended_actions: List[str]
    predicted_time_to_event: Optional[float] = None  # minutes
    historical_precedents: List[Dict[str, Any]] = field(default_factory=list)
    drilling_parameters: Optional[DrillingParameters] = None


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result."""
    assessment_id: str
    timestamp: datetime
    overall_risk_level: RiskLevel
    individual_risks: Dict[DrillingHazard, float]  # Risk probabilities
    risk_factors: List[str]
    drilling_optimization: Dict[str, Any]
    formation_analysis: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


class DrillingDysfunctionDetector:
    """Detect drilling dysfunctions using real-time parameters."""
    
    def __init__(self):
        self.mse_thresholds = {
            "sandstone": 200000,  # psi
            "shale": 300000,
            "limestone": 250000,
            "default": 250000
        }
        
        self.parameter_windows = {
            "torque": deque(maxlen=50),
            "rop": deque(maxlen=50), 
            "wob": deque(maxlen=50),
            "rpm": deque(maxlen=50),
            "pressure": deque(maxlen=50)
        }
    
    def analyze_drilling_efficiency(self, params: DrillingParameters) -> Dict[str, Any]:
        """Analyze drilling efficiency and detect dysfunctions."""
        
        # Update parameter windows
        self.parameter_windows["torque"].append(params.torque)
        self.parameter_windows["rop"].append(params.rate_of_penetration)
        self.parameter_windows["wob"].append(params.weight_on_bit)
        self.parameter_windows["rpm"].append(params.rotary_speed)
        self.parameter_windows["pressure"].append(params.pump_pressure)
        
        dysfunctions = []
        efficiency_metrics = {}
        
        # Calculate MSE
        mse = self._calculate_mse(params)
        efficiency_metrics["mse"] = mse
        
        # MSE-based dysfunction detection
        threshold = self.mse_thresholds.get("default", 250000)
        if mse > threshold:
            dysfunctions.append({
                "type": "high_mse",
                "severity": "warning" if mse < threshold * 1.5 else "critical",
                "description": f"High MSE ({mse:.0f} psi) indicates drilling inefficiency",
                "recommendation": "Optimize WOB and RPM parameters"
            })
        
        # Stick-slip detection
        if len(self.parameter_windows["rpm"]) > 10:
            rpm_data = np.array(self.parameter_windows["rpm"])
            rpm_cv = np.std(rpm_data) / np.mean(rpm_data) if np.mean(rpm_data) > 0 else 0
            
            if rpm_cv > 0.15:  # 15% coefficient of variation
                dysfunctions.append({
                    "type": "stick_slip",
                    "severity": "warning",
                    "description": f"Stick-slip detected (RPM CV: {rpm_cv:.2f})",
                    "recommendation": "Reduce WOB and adjust RPM"
                })
        
        # Whirl detection using torque analysis
        if len(self.parameter_windows["torque"]) > 20:
            torque_data = np.array(self.parameter_windows["torque"])
            torque_peaks, _ = find_peaks(torque_data, height=np.mean(torque_data) * 1.2)
            
            if len(torque_peaks) > len(torque_data) * 0.3:  # More than 30% peaks
                dysfunctions.append({
                    "type": "whirl",
                    "severity": "critical",
                    "description": "Potential whirl detected from torque oscillations",
                    "recommendation": "Reduce RPM and check BHA design"
                })
        
        # ROP efficiency analysis
        if len(self.parameter_windows["rop"]) > 5:
            recent_rop = np.mean(list(self.parameter_windows["rop"])[-5:])
            target_rop = self._calculate_target_rop(params)
            
            efficiency_ratio = recent_rop / max(target_rop, 1.0)
            efficiency_metrics["rop_efficiency"] = efficiency_ratio
            
            if efficiency_ratio < 0.5:
                dysfunctions.append({
                    "type": "low_rop_efficiency",
                    "severity": "warning",
                    "description": f"ROP efficiency at {efficiency_ratio:.1%}",
                    "recommendation": "Review drilling parameters and bit condition"
                })
        
        return {
            "dysfunctions": dysfunctions,
            "efficiency_metrics": efficiency_metrics,
            "optimization_potential": self._assess_optimization_potential(params)
        }
    
    def _calculate_mse(self, params: DrillingParameters) -> float:
        """Calculate Mechanical Specific Energy."""
        if params.rate_of_penetration <= 0:
            return float('inf')
        
        # MSE = (WOB + 4π * Torque * RPM / (12 * ROP)) / Bit Area
        bit_area = np.pi * (8.5 ** 2) / 4  # Assume 8.5" bit
        
        mse = (params.weight_on_bit * 1000 + 
               (4 * np.pi * params.torque * params.rotary_speed) / 
               (12 * params.rate_of_penetration)) / bit_area
        
        return max(mse, 0)
    
    def _calculate_target_rop(self, params: DrillingParameters) -> float:
        """Calculate target ROP based on formation and parameters."""
        # Simplified ROP model (would use more sophisticated models in production)
        base_rop = 20.0  # ft/hr baseline
        
        # Adjust for WOB
        wob_factor = min(params.weight_on_bit / 30.0, 2.0)  # Normalize to 30 klbs
        
        # Adjust for RPM
        rpm_factor = min(params.rotary_speed / 100.0, 1.5)  # Normalize to 100 RPM
        
        target_rop = base_rop * wob_factor * rpm_factor
        
        return max(target_rop, 1.0)
    
    def _assess_optimization_potential(self, params: DrillingParameters) -> Dict[str, Any]:
        """Assess potential for parameter optimization."""
        
        optimizations = {}
        
        # WOB optimization
        current_mse = self._calculate_mse(params)
        
        # Test different WOB values
        wob_range = np.linspace(params.weight_on_bit * 0.8, params.weight_on_bit * 1.2, 5)
        optimal_wob = params.weight_on_bit
        min_mse = current_mse
        
        for test_wob in wob_range:
            test_params = DrillingParameters(
                well_id=params.well_id,
                measured_depth=params.measured_depth,
                weight_on_bit=test_wob,
                rotary_speed=params.rotary_speed,
                rate_of_penetration=params.rate_of_penetration,
                torque=params.torque,
                pump_pressure=params.pump_pressure,
                flow_rate=params.flow_rate,
                mud_weight=params.mud_weight,
                mud_viscosity=params.mud_viscosity,
                mud_temperature=params.mud_temperature
            )
            
            test_mse = self._calculate_mse(test_params)
            if test_mse < min_mse:
                min_mse = test_mse
                optimal_wob = test_wob
        
        optimizations["wob"] = {
            "current": params.weight_on_bit,
            "optimal": optimal_wob,
            "improvement_potential": (current_mse - min_mse) / current_mse if current_mse > 0 else 0
        }
        
        return optimizations


class KickDetectionSystem:
    """Advanced kick detection using multiple indicators."""
    
    def __init__(self):
        self.baseline_params = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
        # Kick indicators thresholds
        self.thresholds = {
            "flow_gain": 0.05,  # 5% flow gain
            "pit_volume_gain": 5.0,  # 5 bbl gain
            "gas_increase": 50.0,  # 50 ppm increase
            "pump_pressure_drop": 50.0,  # 50 psi drop
            "connection_gas": 100.0  # 100 ppm
        }
    
    def initialize_baseline(self, historical_params: List[DrillingParameters]):
        """Initialize baseline parameters from historical data."""
        if not historical_params:
            return
        
        # Calculate baseline statistics
        flow_rates = [p.flow_rate for p in historical_params]
        pressures = [p.pump_pressure for p in historical_params]
        mud_weights = [p.mud_weight for p in historical_params]
        
        self.baseline_params = {
            "flow_rate": {
                "mean": np.mean(flow_rates),
                "std": np.std(flow_rates)
            },
            "pump_pressure": {
                "mean": np.mean(pressures),
                "std": np.std(pressures)
            },
            "mud_weight": {
                "mean": np.mean(mud_weights),
                "std": np.std(mud_weights)
            }
        }
        
        # Train anomaly detection model
        features = []
        for params in historical_params:
            feature_vector = [
                params.flow_rate,
                params.pump_pressure,
                params.mud_weight,
                params.total_gas or 0,
                params.rate_of_penetration
            ]
            features.append(feature_vector)
        
        if len(features) > 10:
            self.anomaly_detector.fit(features)
            self.is_trained = True
    
    def detect_kick_indicators(self, current_params: DrillingParameters) -> Dict[str, Any]:
        """Detect kick indicators from current drilling parameters."""
        
        indicators = {}
        risk_score = 0.0
        
        # Primary kick indicators
        
        # 1. Flow gain detection (would need flow out measurement)
        # For now, use pump pressure drop as proxy
        if "pump_pressure" in self.baseline_params:
            baseline_pressure = self.baseline_params["pump_pressure"]["mean"]
            pressure_drop = baseline_pressure - current_params.pump_pressure
            
            if pressure_drop > self.thresholds["pump_pressure_drop"]:
                indicators["pressure_drop"] = {
                    "detected": True,
                    "value": pressure_drop,
                    "severity": "critical" if pressure_drop > 100 else "warning"
                }
                risk_score += 0.3
        
        # 2. Gas increase detection
        if hasattr(current_params, 'total_gas') and current_params.total_gas:
            if current_params.total_gas > self.thresholds["gas_increase"]:
                indicators["gas_increase"] = {
                    "detected": True,
                    "value": current_params.total_gas,
                    "severity": "critical" if current_params.total_gas > 200 else "warning"
                }
                risk_score += 0.4
        
        # 3. Connection gas (C1-C5 analysis)
        connection_gas = 0
        if hasattr(current_params, 'c1_gas') and current_params.c1_gas:
            connection_gas += current_params.c1_gas
        if hasattr(current_params, 'c2_gas') and current_params.c2_gas:
            connection_gas += current_params.c2_gas
        if hasattr(current_params, 'c3_gas') and current_params.c3_gas:
            connection_gas += current_params.c3_gas
        
        if connection_gas > self.thresholds["connection_gas"]:
            indicators["connection_gas"] = {
                "detected": True,
                "value": connection_gas,
                "severity": "warning"
            }
            risk_score += 0.2
        
        # 4. Drilling break (sudden ROP increase)
        if hasattr(self, 'previous_rop'):
            rop_increase = current_params.rate_of_penetration - self.previous_rop
            if rop_increase > 50:  # 50 ft/hr increase
                indicators["drilling_break"] = {
                    "detected": True,
                    "value": rop_increase,
                    "severity": "warning"
                }
                risk_score += 0.15
        
        self.previous_rop = current_params.rate_of_penetration
        
        # Anomaly detection
        if self.is_trained:
            feature_vector = [[
                current_params.flow_rate,
                current_params.pump_pressure,
                current_params.mud_weight,
                current_params.total_gas or 0,
                current_params.rate_of_penetration
            ]]
            
            anomaly_score = self.anomaly_detector.decision_function(feature_vector)[0]
            is_anomaly = self.anomaly_detector.predict(feature_vector)[0] == -1
            
            if is_anomaly:
                indicators["anomaly_detection"] = {
                    "detected": True,
                    "anomaly_score": anomaly_score,
                    "severity": "warning"
                }
                risk_score += 0.2
        
        # Calculate overall kick probability
        kick_probability = min(risk_score, 1.0)
        
        return {
            "kick_probability": kick_probability,
            "indicators": indicators,
            "overall_assessment": self._assess_kick_risk(kick_probability),
            "recommended_actions": self._get_kick_response_actions(kick_probability)
        }
    
    def _assess_kick_risk(self, probability: float) -> Dict[str, Any]:
        """Assess overall kick risk level."""
        
        if probability >= 0.8:
            risk_level = RiskLevel.CRITICAL
            description = "High probability kick in progress"
        elif probability >= 0.5:
            risk_level = RiskLevel.HIGH
            description = "Elevated kick risk - monitor closely"
        elif probability >= 0.3:
            risk_level = RiskLevel.MEDIUM
            description = "Moderate kick risk - maintain vigilance"
        else:
            risk_level = RiskLevel.LOW
            description = "Normal drilling conditions"
        
        return {
            "risk_level": risk_level,
            "description": description,
            "confidence": min(probability * 1.2, 1.0)
        }
    
    def _get_kick_response_actions(self, probability: float) -> List[str]:
        """Get recommended actions based on kick probability."""
        
        actions = []
        
        if probability >= 0.8:
            actions.extend([
                "STOP DRILLING IMMEDIATELY",
                "Check flow rates and pit levels",
                "Prepare for well control procedures",
                "Alert drilling supervisor",
                "Increase mud weight if confirmed"
            ])
        elif probability >= 0.5:
            actions.extend([
                "Reduce ROP and monitor closely",
                "Increase circulation rate",
                "Check gas readings every 30 seconds",
                "Prepare mud weight increase materials"
            ])
        elif probability >= 0.3:
            actions.extend([
                "Monitor gas readings closely",
                "Check pit volume trends",
                "Maintain current drilling parameters",
                "Brief crew on kick indicators"
            ])
        
        return actions


class LostCirculationPredictor:
    """Predict and detect lost circulation events."""
    
    def __init__(self):
        self.fracture_gradient_model = None
        self.loss_history = []
        
    def predict_loss_zones(self, 
                          formation_data: List[GeologicalFormation],
                          current_params: DrillingParameters) -> Dict[str, Any]:
        """Predict potential lost circulation zones."""
        
        risk_zones = []
        
        for formation in formation_data:
            loss_risk = self._assess_formation_loss_risk(formation, current_params)
            
            if loss_risk["probability"] > 0.3:
                risk_zones.append({
                    "formation": formation.formation_name,
                    "depth_range": (formation.top_depth, formation.bottom_depth),
                    "risk_probability": loss_risk["probability"],
                    "risk_factors": loss_risk["factors"],
                    "severity": loss_risk["severity"],
                    "mitigation_strategies": loss_risk["mitigation"]
                })
        
        # Sort by risk probability
        risk_zones.sort(key=lambda x: x["risk_probability"], reverse=True)
        
        return {
            "total_risk_zones": len(risk_zones),
            "high_risk_zones": len([z for z in risk_zones if z["risk_probability"] > 0.7]),
            "risk_zones": risk_zones[:10],  # Top 10 risk zones
            "overall_loss_risk": self._calculate_overall_loss_risk(risk_zones),
            "mud_weight_recommendations": self._get_mud_weight_recommendations(risk_zones, current_params)
        }
    
    def _assess_formation_loss_risk(self, 
                                   formation: GeologicalFormation, 
                                   current_params: DrillingParameters) -> Dict[str, Any]:
        """Assess lost circulation risk for a specific formation."""
        
        risk_factors = []
        base_probability = 0.1  # 10% base risk
        
        # Lithology-based risk
        high_risk_lithologies = [
            LithologyType.LIMESTONE,  # Naturally fractured
            LithologyType.DOLOMITE,   # Vugs and cavities
        ]
        
        if formation.primary_lithology in high_risk_lithologies:
            risk_factors.append(f"High-risk lithology: {formation.primary_lithology.value}")
            base_probability += 0.3
        
        # Formation thickness risk
        if formation.thickness > 500:  # Thick formations
            risk_factors.append("Thick formation interval")
            base_probability += 0.1
        
        # Drilling hazards
        if DrillingHazard.LOST_CIRCULATION in formation.drilling_hazards:
            risk_factors.append("Historical lost circulation in formation")
            base_probability += 0.4
        
        # Mud weight vs. fracture gradient
        estimated_fracture_gradient = self._estimate_fracture_gradient(formation)
        equivalent_mud_weight = current_params.mud_weight
        
        safety_margin = estimated_fracture_gradient - equivalent_mud_weight * 0.052  # Convert ppg to psi/ft
        
        if safety_margin < 0.5:  # Less than 0.5 psi/ft safety margin
            risk_factors.append(f"Low safety margin: {safety_margin:.2f} psi/ft")
            base_probability += 0.3
        elif safety_margin < 1.0:
            risk_factors.append(f"Moderate safety margin: {safety_margin:.2f} psi/ft")
            base_probability += 0.15
        
        # Pressure while drilling
        if current_params.pump_pressure > 3000:  # High pressure
            risk_factors.append("High pump pressure")
            base_probability += 0.1
        
        final_probability = min(base_probability, 1.0)
        
        # Determine severity
        if final_probability > 0.7:
            severity = "critical"
        elif final_probability > 0.4:
            severity = "high"
        elif final_probability > 0.2:
            severity = "moderate"
        else:
            severity = "low"
        
        # Mitigation strategies
        mitigation = self._get_loss_mitigation_strategies(risk_factors, formation)
        
        return {
            "probability": final_probability,
            "factors": risk_factors,
            "severity": severity,
            "mitigation": mitigation,
            "fracture_gradient": estimated_fracture_gradient,
            "safety_margin": safety_margin
        }
    
    def _estimate_fracture_gradient(self, formation: GeologicalFormation) -> float:
        """Estimate fracture gradient for formation (psi/ft)."""
        
        # Simplified fracture gradient estimation
        depth = (formation.top_depth + formation.bottom_depth) / 2
        
        # Base gradient (varies by geology)
        if formation.primary_lithology == LithologyType.SHALE:
            base_gradient = 0.8  # psi/ft
        elif formation.primary_lithology in [LithologyType.SANDSTONE]:
            base_gradient = 0.75
        elif formation.primary_lithology in [LithologyType.LIMESTONE, LithologyType.DOLOMITE]:
            base_gradient = 0.85
        else:
            base_gradient = 0.8
        
        # Depth adjustment (higher stress with depth)
        depth_factor = 1 + (depth - 5000) / 50000  # Increase with depth
        
        fracture_gradient = base_gradient * depth_factor
        
        return max(fracture_gradient, 0.6)  # Minimum reasonable value
    
    def _calculate_overall_loss_risk(self, risk_zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall lost circulation risk."""
        
        if not risk_zones:
            return {"risk_level": RiskLevel.LOW, "probability": 0.0}
        
        # Weighted average by formation thickness
        total_weighted_risk = 0
        total_thickness = 0
        
        for zone in risk_zones:
            thickness = zone["depth_range"][1] - zone["depth_range"][0]
            total_weighted_risk += zone["risk_probability"] * thickness
            total_thickness += thickness
        
        overall_probability = total_weighted_risk / max(total_thickness, 1)
        
        if overall_probability > 0.7:
            risk_level = RiskLevel.CRITICAL
        elif overall_probability > 0.4:
            risk_level = RiskLevel.HIGH
        elif overall_probability > 0.2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            "risk_level": risk_level,
            "probability": overall_probability,
            "confidence": min(len(risk_zones) / 5.0, 1.0)  # More zones = higher confidence
        }
    
    def _get_mud_weight_recommendations(self, 
                                       risk_zones: List[Dict[str, Any]], 
                                       current_params: DrillingParameters) -> Dict[str, Any]:
        """Get mud weight recommendations to prevent losses."""
        
        recommendations = {
            "current_mud_weight": current_params.mud_weight,
            "recommended_changes": []
        }
        
        for zone in risk_zones[:3]:  # Top 3 risk zones
            if zone["risk_probability"] > 0.5:
                recommendations["recommended_changes"].append({
                    "depth_range": zone["depth_range"],
                    "formation": zone["formation"],
                    "action": "reduce_mud_weight",
                    "suggested_reduction": "0.2-0.5 ppg",
                    "rationale": "High loss risk - reduce ECD"
                })
        
        return recommendations
    
    def _get_loss_mitigation_strategies(self, 
                                       risk_factors: List[str], 
                                       formation: GeologicalFormation) -> List[str]:
        """Get mitigation strategies for lost circulation."""
        
        strategies = []
        
        if "High-risk lithology" in str(risk_factors):
            strategies.extend([
                "Use sized calcium carbonate (CaCO3) as bridging agent",
                "Prepare lost circulation material (LCM) pills",
                "Consider drilling with managed pressure drilling (MPD)"
            ])
        
        if "Low safety margin" in str(risk_factors):
            strategies.extend([
                "Reduce mud weight within safe limits",
                "Optimize drilling hydraulics to reduce ECD",
                "Use low-rheology mud system"
            ])
        
        if "Thick formation" in str(risk_factors):
            strategies.extend([
                "Prepare multiple LCM pill volumes",
                "Have cement available for severe losses",
                "Consider casing point adjustment"
            ])
        
        # Formation-specific strategies
        if formation.primary_lithology == LithologyType.LIMESTONE:
            strategies.append("Use resilient graphitic carbon (RGC) for sealing fractures")
        
        return strategies


class RealTimeRiskMonitor:
    """Real-time monitoring and alert system for drilling risks."""
    
    def __init__(self):
        self.kick_detector = KickDetectionSystem()
        self.loss_predictor = LostCirculationPredictor()
        self.dysfunction_detector = DrillingDysfunctionDetector()
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Real-time data buffers
        self.parameter_buffer = deque(maxlen=100)
        
    async def process_real_time_data(self, params: DrillingParameters) -> Dict[str, Any]:
        """Process real-time drilling parameters for risk assessment."""
        
        # Add to buffer
        self.parameter_buffer.append(params)
        
        # Comprehensive risk analysis
        risk_analysis = {
            "timestamp": params.timestamp,
            "depth": params.measured_depth,
            "kick_analysis": {},
            "dysfunction_analysis": {},
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # 1. Kick detection analysis
            kick_analysis = self.kick_detector.detect_kick_indicators(params)
            risk_analysis["kick_analysis"] = kick_analysis
            
            # Generate kick alerts
            if kick_analysis["kick_probability"] > 0.5:
                alert = RiskAlert(
                    alert_id=f"kick_{int(params.timestamp.timestamp())}",
                    timestamp=params.timestamp,
                    risk_type=DrillingHazard.KICK,
                    severity=AlertSeverity.CRITICAL if kick_analysis["kick_probability"] > 0.8 else AlertSeverity.WARNING,
                    category=RiskCategory.FORMATION,
                    probability=kick_analysis["kick_probability"],
                    confidence=kick_analysis["overall_assessment"]["confidence"],
                    current_depth=params.measured_depth,
                    description=kick_analysis["overall_assessment"]["description"],
                    recommended_actions=kick_analysis["recommended_actions"],
                    drilling_parameters=params
                )
                
                risk_analysis["alerts"].append(alert)
                await self._process_alert(alert)
            
            # 2. Drilling dysfunction analysis
            dysfunction_analysis = self.dysfunction_detector.analyze_drilling_efficiency(params)
            risk_analysis["dysfunction_analysis"] = dysfunction_analysis
            
            # Generate dysfunction alerts
            for dysfunction in dysfunction_analysis["dysfunctions"]:
                if dysfunction["severity"] in ["warning", "critical"]:
                    hazard_type = self._map_dysfunction_to_hazard(dysfunction["type"])
                    
                    alert = RiskAlert(
                        alert_id=f"{dysfunction['type']}_{int(params.timestamp.timestamp())}",
                        timestamp=params.timestamp,
                        risk_type=hazard_type,
                        severity=AlertSeverity.CRITICAL if dysfunction["severity"] == "critical" else AlertSeverity.WARNING,
                        category=RiskCategory.MECHANICAL,
                        probability=0.8 if dysfunction["severity"] == "critical" else 0.5,
                        confidence=0.7,
                        current_depth=params.measured_depth,
                        description=dysfunction["description"],
                        recommended_actions=[dysfunction["recommendation"]],
                        drilling_parameters=params
                    )
                    
                    risk_analysis["alerts"].append(alert)
                    await self._process_alert(alert)
            
            # 3. Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                kick_analysis, dysfunction_analysis, params
            )
            risk_analysis["recommendations"] = recommendations
            
        except Exception as e:
            logger.error(f"Risk monitoring failed: {e}")
            risk_analysis["error"] = str(e)
        
        return risk_analysis
    
    async def _process_alert(self, alert: RiskAlert):
        """Process and manage alerts."""
        
        # Check for duplicate alerts
        alert_key = f"{alert.risk_type}_{alert.current_depth:.0f}"
        
        if alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[alert_key]
            if alert.probability > existing_alert.probability:
                self.active_alerts[alert_key] = alert
        else:
            # New alert
            self.active_alerts[alert_key] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"Drilling Alert: {alert.risk_type.value} at {alert.current_depth} ft - {alert.description}")
    
    def _map_dysfunction_to_hazard(self, dysfunction_type: str) -> DrillingHazard:
        """Map dysfunction type to drilling hazard."""
        
        mapping = {
            "high_mse": DrillingHazard.STUCK_PIPE,
            "stick_slip": DrillingHazard.TWIST_OFF,
            "whirl": DrillingHazard.TWIST_OFF,
            "low_rop_efficiency": DrillingHazard.TIGHT_HOLE
        }
        
        return mapping.get(dysfunction_type, DrillingHazard.STUCK_PIPE)
    
    def _generate_optimization_recommendations(self, 
                                            kick_analysis: Dict[str, Any],
                                            dysfunction_analysis: Dict[str, Any],
                                            params: DrillingParameters) -> List[str]:
        """Generate drilling parameter optimization recommendations."""
        
        recommendations = []
        
        # Kick-based recommendations
        if kick_analysis["kick_probability"] > 0.3:
            recommendations.append("Increase mud weight by 0.2-0.5 ppg")
            recommendations.append("Reduce ROP to improve hole cleaning")
        
        # Dysfunction-based recommendations
        for dysfunction in dysfunction_analysis["dysfunctions"]:
            if dysfunction["type"] == "high_mse":
                recommendations.append("Optimize WOB/RPM ratio to reduce MSE")
            elif dysfunction["type"] == "stick_slip":
                recommendations.append("Reduce WOB and increase RPM to minimize stick-slip")
            elif dysfunction["type"] == "whirl":
                recommendations.append("Reduce RPM and review BHA design")
        
        # Efficiency-based recommendations
        efficiency_metrics = dysfunction_analysis.get("efficiency_metrics", {})
        if efficiency_metrics.get("rop_efficiency", 1.0) < 0.7:
            recommendations.append("Consider bit change or parameter adjustment")
        
        # Optimization potential
        optimization = dysfunction_analysis.get("optimization_potential", {})
        if "wob" in optimization and optimization["wob"]["improvement_potential"] > 0.1:
            optimal_wob = optimization["wob"]["optimal"]
            recommendations.append(f"Adjust WOB to {optimal_wob:.1f} klbs for optimal MSE")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def clear_alert(self, alert_id: str):
        """Clear a specific alert."""
        for key, alert in list(self.active_alerts.items()):
            if alert.alert_id == alert_id:
                del self.active_alerts[key]
                break
    
    def get_alert_history(self, hours: int = 24) -> List[RiskAlert]:
        """Get alert history for specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [alert for alert in self.alert_history 
                if alert.timestamp > cutoff_time]


class DrillingRiskPredictor:
    """Main drilling risk prediction engine."""
    
    def __init__(self):
        self.real_time_monitor = RealTimeRiskMonitor()
        self.risk_models = {}
        self.formation_database = {}
        
    async def initialize_models(self, 
                               historical_data: List[DrillingParameters],
                               formation_data: List[GeologicalFormation]):
        """Initialize prediction models with historical data."""
        
        # Initialize kick detector baseline
        self.real_time_monitor.kick_detector.initialize_baseline(historical_data)
        
        # Store formation data
        for formation in formation_data:
            self.formation_database[formation.formation_name] = formation
        
        logger.info(f"Initialized drilling risk models with {len(historical_data)} historical records")
    
    async def assess_comprehensive_risk(self, 
                                       current_params: DrillingParameters,
                                       upcoming_formations: List[GeologicalFormation] = None) -> RiskAssessment:
        """Comprehensive risk assessment for current conditions."""
        
        assessment_id = f"risk_assess_{int(current_params.timestamp.timestamp())}"
        
        # Real-time analysis
        real_time_analysis = await self.real_time_monitor.process_real_time_data(current_params)
        
        # Formation-based risk analysis
        formation_risks = {}
        if upcoming_formations:
            formation_risks = self._assess_formation_risks(upcoming_formations, current_params)
        
        # Combine all risk factors
        individual_risks = {
            DrillingHazard.KICK: real_time_analysis["kick_analysis"].get("kick_probability", 0.0),
            DrillingHazard.STUCK_PIPE: self._calculate_stuck_pipe_risk(current_params),
            DrillingHazard.LOST_CIRCULATION: formation_risks.get("lost_circulation", {}).get("probability", 0.0),
            DrillingHazard.WASHOUT: self._calculate_washout_risk(current_params),
            DrillingHazard.TIGHT_HOLE: self._calculate_tight_hole_risk(current_params)
        }
        
        # Overall risk level
        max_risk = max(individual_risks.values())
        if max_risk > 0.8:
            overall_risk = RiskLevel.CRITICAL
        elif max_risk > 0.5:
            overall_risk = RiskLevel.HIGH
        elif max_risk > 0.3:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        # Risk factors summary
        risk_factors = []
        for hazard, probability in individual_risks.items():
            if probability > 0.3:
                risk_factors.append(f"{hazard.value}: {probability:.1%} probability")
        
        # Drilling optimization recommendations
        drilling_optimization = {
            "current_parameters": {
                "wob": current_params.weight_on_bit,
                "rpm": current_params.rotary_speed,
                "flow_rate": current_params.flow_rate,
                "mud_weight": current_params.mud_weight
            },
            "optimization_potential": real_time_analysis["dysfunction_analysis"].get("optimization_potential", {}),
            "efficiency_metrics": real_time_analysis["dysfunction_analysis"].get("efficiency_metrics", {})
        }
        
        # Confidence score
        confidence_score = self._calculate_confidence_score(real_time_analysis, formation_risks)
        
        return RiskAssessment(
            assessment_id=assessment_id,
            timestamp=current_params.timestamp,
            overall_risk_level=overall_risk,
            individual_risks=individual_risks,
            risk_factors=risk_factors,
            drilling_optimization=drilling_optimization,
            formation_analysis=formation_risks,
            recommendations=real_time_analysis.get("recommendations", []),
            confidence_score=confidence_score
        )
    
    def _assess_formation_risks(self, 
                               formations: List[GeologicalFormation],
                               current_params: DrillingParameters) -> Dict[str, Any]:
        """Assess risks from upcoming formations."""
        
        formation_analysis = {}
        
        # Lost circulation analysis
        loss_analysis = self.real_time_monitor.loss_predictor.predict_loss_zones(
            formations, current_params
        )
        formation_analysis["lost_circulation"] = loss_analysis.get("overall_loss_risk", {})
        
        # Wellbore stability analysis
        stability_risks = []
        for formation in formations:
            if formation.primary_lithology == LithologyType.SHALE:
                stability_risks.append({
                    "formation": formation.formation_name,
                    "depth": formation.top_depth,
                    "risk": "shale_instability",
                    "probability": 0.4 if formation.thickness > 200 else 0.2
                })
        
        formation_analysis["wellbore_stability"] = stability_risks
        
        return formation_analysis
    
    def _calculate_stuck_pipe_risk(self, params: DrillingParameters) -> float:
        """Calculate stuck pipe risk based on current parameters."""
        
        risk_factors = 0.0
        
        # High MSE increases stuck pipe risk
        mse = params.calculate_mse()
        if mse > 300000:
            risk_factors += 0.3
        elif mse > 200000:
            risk_factors += 0.15
        
        # Low ROP with high WOB
        if params.rate_of_penetration < 15 and params.weight_on_bit > 40:
            risk_factors += 0.2
        
        # High torque
        if params.torque > 25000:
            risk_factors += 0.15
        
        # Poor hole cleaning (high mud weight, low flow rate)
        if params.mud_weight > 12 and params.flow_rate < 300:
            risk_factors += 0.1
        
        return min(risk_factors, 1.0)
    
    def _calculate_washout_risk(self, params: DrillingParameters) -> float:
        """Calculate washout risk."""
        
        risk_factors = 0.0
        
        # High pump pressure
        if params.pump_pressure > 4000:
            risk_factors += 0.2
        
        # High flow rate
        if params.flow_rate > 800:
            risk_factors += 0.15
        
        # Low mud weight (erosive)
        if params.mud_weight < 9:
            risk_factors += 0.1
        
        return min(risk_factors, 1.0)
    
    def _calculate_tight_hole_risk(self, params: DrillingParameters) -> float:
        """Calculate tight hole risk."""
        
        risk_factors = 0.0
        
        # Low ROP consistently
        if params.rate_of_penetration < 10:
            risk_factors += 0.2
        
        # High torque and drag
        if params.torque > 20000:
            risk_factors += 0.15
        
        # Poor hole cleaning
        if params.flow_rate < 250:
            risk_factors += 0.1
        
        return min(risk_factors, 1.0)
    
    def _calculate_confidence_score(self, 
                                   real_time_analysis: Dict[str, Any],
                                   formation_risks: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the assessment."""
        
        confidence_factors = []
        
        # Real-time data quality
        if "kick_analysis" in real_time_analysis:
            kick_confidence = real_time_analysis["kick_analysis"].get("overall_assessment", {}).get("confidence", 0.5)
            confidence_factors.append(kick_confidence)
        
        # Model training status
        if self.real_time_monitor.kick_detector.is_trained:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Formation data availability
        if formation_risks:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5


# Global instances
drilling_risk_ai = DrillingRiskPredictor()
real_time_monitor = RealTimeRiskMonitor()
kick_detector = KickDetectionSystem()
loss_predictor = LostCirculationPredictor()