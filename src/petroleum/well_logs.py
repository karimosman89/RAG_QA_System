"""
AI-Powered Well Log Interpretation System

Advanced machine learning models for automated well log analysis:
- Multi-log lithology classification using ensemble methods
- Fluid contact detection with uncertainty quantification  
- Formation evaluation and petrophysical property estimation
- Automated log quality control and outlier detection
- Real-time log interpretation during drilling
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import logging

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Signal processing
from scipy import signal
from scipy.stats import zscore
from scipy.interpolate import interp1d

# Deep learning (placeholder - would use TensorFlow/PyTorch in production)
try:
    import tensorflow as tf
except ImportError:
    tf = None

from .data_models import (
    WellLogData, LithologyType, FluidType, LogType,
    GeologicalFormation, petroleum_data_processor
)

logger = logging.getLogger(__name__)


class InterpretationMethod(str, Enum):
    """Log interpretation methods."""
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning" 
    PETROPHYSICAL = "petrophysical"
    STATISTICAL = "statistical"
    HYBRID = "hybrid"


class LogQuality(str, Enum):
    """Log data quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class InterpretationResult:
    """Well log interpretation result."""
    depth: float
    lithology: LithologyType
    confidence: float
    porosity: Optional[float] = None
    permeability: Optional[float] = None
    water_saturation: Optional[float] = None
    fluid_type: Optional[FluidType] = None
    formation_name: Optional[str] = None
    interpretation_method: InterpretationMethod = InterpretationMethod.MACHINE_LEARNING
    quality_flag: LogQuality = LogQuality.GOOD
    metadata: Dict[str, Any] = None


class LithologyClassifier:
    """Advanced ML-based lithology classification."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            ),
            'gradient_boost': None  # Will be initialized when needed
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Lithology encoding
        self.lithology_encoder = {
            LithologyType.SANDSTONE: 0,
            LithologyType.SHALE: 1,
            LithologyType.LIMESTONE: 2,
            LithologyType.DOLOMITE: 3,
            LithologyType.COAL: 4,
            LithologyType.SALT: 5
        }
        self.lithology_decoder = {v: k for k, v in self.lithology_encoder.items()}
    
    def prepare_features(self, logs_data: Dict[LogType, WellLogData]) -> np.ndarray:
        """Prepare feature matrix from well logs."""
        features = []
        feature_names = []
        
        # Common depth array (interpolate all logs to same depths)
        if LogType.GAMMA_RAY in logs_data:
            reference_depths = np.array(logs_data[LogType.GAMMA_RAY].depths)
        else:
            # Use first available log as reference
            reference_depths = np.array(list(logs_data.values())[0].depths)
        
        # Extract and interpolate log values
        for log_type in [LogType.GAMMA_RAY, LogType.RESISTIVITY, LogType.NEUTRON, 
                        LogType.DENSITY, LogType.PHOTOELECTRIC, LogType.SP]:
            if log_type in logs_data:
                log_data = logs_data[log_type]
                
                # Interpolate to reference depths
                if len(log_data.depths) != len(reference_depths):
                    interp_func = interp1d(
                        log_data.depths, log_data.values, 
                        kind='linear', fill_value='extrapolate'
                    )
                    values = interp_func(reference_depths)
                else:
                    values = np.array(log_data.values)
                
                features.append(values)
                feature_names.append(log_type.value)
                
                # Add derived features
                if log_type == LogType.GAMMA_RAY:
                    # Add gamma ray derivatives
                    gr_smooth = signal.savgol_filter(values, 5, 2)
                    gr_gradient = np.gradient(gr_smooth)
                    features.extend([gr_smooth, gr_gradient])
                    feature_names.extend(['gr_smooth', 'gr_gradient'])
                
                elif log_type == LogType.RESISTIVITY:
                    # Add resistivity ratios and logs
                    log_res = np.log10(np.maximum(values, 0.1))  # Avoid log(0)
                    features.append(log_res)
                    feature_names.append('log_resistivity')
        
        # Add composite features
        if LogType.NEUTRON in logs_data and LogType.DENSITY in logs_data:
            neutron_values = np.array(logs_data[LogType.NEUTRON].values)
            density_values = np.array(logs_data[LogType.DENSITY].values)
            
            # Neutron-density separation (gas effect indicator)
            nd_separation = neutron_values - (2.65 - density_values) * 0.45
            features.append(nd_separation)
            feature_names.append('neutron_density_separation')
        
        if LogType.PHOTOELECTRIC in logs_data and LogType.DENSITY in logs_data:
            pe_values = np.array(logs_data[LogType.PHOTOELECTRIC].values)
            density_values = np.array(logs_data[LogType.DENSITY].values)
            
            # Photoelectric factor * density (lithology indicator)
            pe_density = pe_values * density_values
            features.append(pe_density)
            feature_names.append('pe_density_product')
        
        self.feature_names = feature_names
        return np.column_stack(features) if features else np.array([]).reshape(0, 0)
    
    def train_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train lithology classification models."""
        if not training_data:
            return {"error": "No training data provided"}
        
        try:
            # Prepare training features and labels
            X_list = []
            y_list = []
            
            for sample in training_data:
                logs_data = sample.get('logs', {})
                lithology = sample.get('lithology')
                
                if logs_data and lithology in self.lithology_encoder:
                    features = self.prepare_features(logs_data)
                    if features.size > 0:
                        X_list.append(features)
                        y_list.extend([self.lithology_encoder[lithology]] * len(features))
            
            if not X_list:
                return {"error": "No valid training features extracted"}
            
            # Stack all features
            X = np.vstack(X_list)
            y = np.array(y_list)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split training data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest
            self.models['random_forest'].fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.models['random_forest'].score(X_train, y_train)
            test_score = self.models['random_forest'].score(X_test, y_test)
            
            y_pred = self.models['random_forest'].predict(X_test)
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.models['random_forest'].feature_importances_
            ))
            
            self.is_trained = True
            
            return {
                "success": True,
                "training_samples": len(X),
                "training_accuracy": train_score,
                "test_accuracy": test_score,
                "feature_importance": feature_importance,
                "classification_report": classification_report(
                    y_test, y_pred, 
                    target_names=[lith.value for lith in self.lithology_decoder.values()],
                    output_dict=True
                )
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": f"Training failed: {str(e)}"}
    
    def predict_lithology(self, logs_data: Dict[LogType, WellLogData]) -> List[InterpretationResult]:
        """Predict lithology for well logs."""
        if not self.is_trained:
            # Use rule-based classification as fallback
            return self._rule_based_classification(logs_data)
        
        try:
            features = self.prepare_features(logs_data)
            if features.size == 0:
                return []
            
            # Get reference depths
            reference_depths = np.array(list(logs_data.values())[0].depths)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict with Random Forest
            predictions = self.models['random_forest'].predict(features_scaled)
            probabilities = self.models['random_forest'].predict_proba(features_scaled)
            
            results = []
            for i, (depth, pred_class) in enumerate(zip(reference_depths, predictions)):
                lithology = self.lithology_decoder.get(pred_class, LithologyType.UNKNOWN)
                confidence = np.max(probabilities[i]) if len(probabilities[i]) > 0 else 0.5
                
                result = InterpretationResult(
                    depth=depth,
                    lithology=lithology,
                    confidence=confidence,
                    interpretation_method=InterpretationMethod.MACHINE_LEARNING,
                    metadata={
                        "probabilities": dict(zip(
                            [self.lithology_decoder[j] for j in range(len(probabilities[i]))],
                            probabilities[i]
                        ))
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Lithology prediction failed: {e}")
            return self._rule_based_classification(logs_data)
    
    def _rule_based_classification(self, logs_data: Dict[LogType, WellLogData]) -> List[InterpretationResult]:
        """Fallback rule-based lithology classification."""
        results = []
        
        if LogType.GAMMA_RAY not in logs_data:
            return results
        
        gr_log = logs_data[LogType.GAMMA_RAY]
        depths = gr_log.depths
        gr_values = np.array(gr_log.values)
        
        # Simple rules based on gamma ray
        for depth, gr in zip(depths, gr_values):
            if gr < 50:
                lithology = LithologyType.SANDSTONE
                confidence = 0.6
            elif gr > 150:
                lithology = LithologyType.SHALE
                confidence = 0.7
            elif 50 <= gr <= 80:
                lithology = LithologyType.LIMESTONE
                confidence = 0.5
            else:
                lithology = LithologyType.UNKNOWN
                confidence = 0.3
            
            # Refine with resistivity if available
            if LogType.RESISTIVITY in logs_data:
                res_log = logs_data[LogType.RESISTIVITY]
                # Simple interpolation to get resistivity at this depth
                res_interp = np.interp(depth, res_log.depths, res_log.values)
                
                if res_interp > 100 and gr < 80:
                    lithology = LithologyType.LIMESTONE
                    confidence = 0.8
            
            result = InterpretationResult(
                depth=depth,
                lithology=lithology,
                confidence=confidence,
                interpretation_method=InterpretationMethod.PETROPHYSICAL
            )
            results.append(result)
        
        return results


class FluidContactDetector:
    """Advanced fluid contact detection using multiple log signatures."""
    
    def __init__(self):
        self.detection_methods = [
            'resistivity_gradient',
            'neutron_density_crossover',
            'photoelectric_signature',
            'pressure_gradient'
        ]
    
    def detect_contacts(self, logs_data: Dict[LogType, WellLogData]) -> List[Dict[str, Any]]:
        """Detect fluid contacts using multiple methods."""
        contacts = []
        
        # Method 1: Resistivity gradient analysis
        if LogType.RESISTIVITY in logs_data:
            res_contacts = self._resistivity_gradient_method(logs_data[LogType.RESISTIVITY])
            contacts.extend(res_contacts)
        
        # Method 2: Neutron-Density crossover
        if LogType.NEUTRON in logs_data and LogType.DENSITY in logs_data:
            nd_contacts = self._neutron_density_method(
                logs_data[LogType.NEUTRON], logs_data[LogType.DENSITY]
            )
            contacts.extend(nd_contacts)
        
        # Method 3: Photoelectric factor changes
        if LogType.PHOTOELECTRIC in logs_data:
            pe_contacts = self._photoelectric_method(logs_data[LogType.PHOTOELECTRIC])
            contacts.extend(pe_contacts)
        
        # Consolidate and rank contacts
        consolidated_contacts = self._consolidate_contacts(contacts)
        
        return consolidated_contacts
    
    def _resistivity_gradient_method(self, res_log: WellLogData) -> List[Dict[str, Any]]:
        """Detect contacts using resistivity gradient analysis."""
        contacts = []
        
        depths = np.array(res_log.depths)
        values = np.array(res_log.values)
        
        # Calculate gradient
        gradient = np.gradient(values, depths)
        
        # Smooth gradient to reduce noise
        smooth_gradient = signal.savgol_filter(gradient, 11, 2)
        
        # Find significant gradient changes
        gradient_threshold = np.percentile(np.abs(smooth_gradient), 95)
        
        for i in range(1, len(smooth_gradient) - 1):
            if abs(smooth_gradient[i]) > gradient_threshold:
                # Determine contact type
                if smooth_gradient[i] > 0:
                    contact_type = "oil_water"  # Resistivity increases downward
                    fluid_above = FluidType.WATER
                    fluid_below = FluidType.OIL
                else:
                    contact_type = "gas_oil"  # Resistivity decreases downward
                    fluid_above = FluidType.GAS
                    fluid_below = FluidType.OIL
                
                confidence = min(abs(smooth_gradient[i]) / gradient_threshold, 1.0) * 0.8
                
                contacts.append({
                    "depth": depths[i],
                    "contact_type": contact_type,
                    "fluid_above": fluid_above,
                    "fluid_below": fluid_below,
                    "method": "resistivity_gradient",
                    "confidence": confidence,
                    "gradient_value": smooth_gradient[i],
                    "resistivity_above": values[i-1],
                    "resistivity_below": values[i+1]
                })
        
        return contacts
    
    def _neutron_density_method(self, neutron_log: WellLogData, density_log: WellLogData) -> List[Dict[str, Any]]:
        """Detect gas effects using neutron-density crossover."""
        contacts = []
        
        # Interpolate to common depth grid
        common_depths = np.linspace(
            max(min(neutron_log.depths), min(density_log.depths)),
            min(max(neutron_log.depths), max(density_log.depths)),
            min(len(neutron_log.depths), len(density_log.depths))
        )
        
        neutron_interp = np.interp(common_depths, neutron_log.depths, neutron_log.values)
        density_interp = np.interp(common_depths, density_log.depths, density_log.values)
        
        # Calculate neutron-density separation
        nd_separation = neutron_interp - (2.65 - density_interp) * 0.45
        
        # Detect gas effects (negative separation)
        gas_threshold = -0.05  # Neutron porosity units
        
        for i in range(1, len(nd_separation) - 1):
            if (nd_separation[i] < gas_threshold and 
                nd_separation[i-1] > gas_threshold):
                # Gas contact detected
                contacts.append({
                    "depth": common_depths[i],
                    "contact_type": "gas_contact",
                    "fluid_above": FluidType.OIL,
                    "fluid_below": FluidType.GAS,
                    "method": "neutron_density",
                    "confidence": min(abs(nd_separation[i]) / abs(gas_threshold), 1.0) * 0.7,
                    "nd_separation": nd_separation[i],
                    "neutron_porosity": neutron_interp[i],
                    "bulk_density": density_interp[i]
                })
        
        return contacts
    
    def _photoelectric_method(self, pe_log: WellLogData) -> List[Dict[str, Any]]:
        """Detect lithology and fluid changes using photoelectric factor."""
        contacts = []
        
        depths = np.array(pe_log.depths)
        values = np.array(pe_log.values)
        
        # Look for significant PE changes
        pe_changes = np.abs(np.diff(values))
        change_threshold = np.percentile(pe_changes, 90)
        
        for i in range(len(pe_changes)):
            if pe_changes[i] > change_threshold:
                # Classify change type
                pe_above = values[i]
                pe_below = values[i + 1]
                
                contact_info = {
                    "depth": (depths[i] + depths[i + 1]) / 2,
                    "method": "photoelectric",
                    "confidence": min(pe_changes[i] / change_threshold, 1.0) * 0.6,
                    "pe_above": pe_above,
                    "pe_below": pe_below
                }
                
                # Interpret PE change
                if pe_above > 2.8 and pe_below < 2.0:
                    contact_info.update({
                        "contact_type": "limestone_to_sandstone",
                        "lithology_above": LithologyType.LIMESTONE,
                        "lithology_below": LithologyType.SANDSTONE
                    })
                elif pe_above < 2.0 and pe_below > 2.8:
                    contact_info.update({
                        "contact_type": "sandstone_to_limestone", 
                        "lithology_above": LithologyType.SANDSTONE,
                        "lithology_below": LithologyType.LIMESTONE
                    })
                
                contacts.append(contact_info)
        
        return contacts
    
    def _consolidate_contacts(self, contacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate and rank detected contacts."""
        if not contacts:
            return []
        
        # Sort by depth
        contacts.sort(key=lambda x: x["depth"])
        
        # Merge contacts that are close together (within 5 feet)
        consolidated = []
        merge_distance = 5.0
        
        i = 0
        while i < len(contacts):
            current_contact = contacts[i].copy()
            merged_methods = [current_contact["method"]]
            total_confidence = current_contact["confidence"]
            
            # Look for nearby contacts to merge
            j = i + 1
            while j < len(contacts) and contacts[j]["depth"] - current_contact["depth"] <= merge_distance:
                merged_methods.append(contacts[j]["method"])
                total_confidence += contacts[j]["confidence"]
                j += 1
            
            # Average confidence and update metadata
            current_contact["confidence"] = total_confidence / len(merged_methods)
            current_contact["methods_used"] = merged_methods
            current_contact["detection_count"] = len(merged_methods)
            
            consolidated.append(current_contact)
            i = j
        
        # Sort by confidence (highest first)
        consolidated.sort(key=lambda x: x["confidence"], reverse=True)
        
        return consolidated


class LogAnalysisEngine:
    """Comprehensive well log analysis engine."""
    
    def __init__(self):
        self.lithology_classifier = LithologyClassifier()
        self.fluid_detector = FluidContactDetector()
        self.quality_analyzer = LogQualityAnalyzer()
        
    def analyze_well_logs(
        self, 
        logs_data: Dict[LogType, WellLogData],
        analysis_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Complete well log analysis."""
        
        if not logs_data:
            return {"error": "No log data provided"}
        
        options = analysis_options or {}
        
        analysis_results = {
            "well_id": list(logs_data.values())[0].well_id,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "available_logs": list(logs_data.keys()),
            "depth_range": self._get_depth_range(logs_data),
            "log_quality": {},
            "lithology_interpretation": [],
            "fluid_contacts": [],
            "petrophysical_properties": {},
            "formation_boundaries": [],
            "recommendations": []
        }
        
        try:
            # 1. Log Quality Assessment
            for log_type, log_data in logs_data.items():
                quality_result = self.quality_analyzer.assess_quality(log_data)
                analysis_results["log_quality"][log_type.value] = quality_result
            
            # 2. Lithology Interpretation
            if options.get("perform_lithology", True):
                lithology_results = self.lithology_classifier.predict_lithology(logs_data)
                analysis_results["lithology_interpretation"] = [
                    {
                        "depth": r.depth,
                        "lithology": r.lithology.value,
                        "confidence": r.confidence,
                        "method": r.interpretation_method.value
                    }
                    for r in lithology_results
                ]
            
            # 3. Fluid Contact Detection
            if options.get("detect_contacts", True):
                contacts = self.fluid_detector.detect_contacts(logs_data)
                analysis_results["fluid_contacts"] = contacts
            
            # 4. Petrophysical Property Estimation
            if options.get("calculate_properties", True):
                properties = self._calculate_petrophysical_properties(logs_data)
                analysis_results["petrophysical_properties"] = properties
            
            # 5. Formation Boundary Detection
            if options.get("detect_formations", True):
                boundaries = self._detect_formation_boundaries(logs_data)
                analysis_results["formation_boundaries"] = boundaries
            
            # 6. Generate Recommendations
            recommendations = self._generate_recommendations(analysis_results)
            analysis_results["recommendations"] = recommendations
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _get_depth_range(self, logs_data: Dict[LogType, WellLogData]) -> Dict[str, float]:
        """Get overall depth range for all logs."""
        all_depths = []
        for log_data in logs_data.values():
            all_depths.extend(log_data.depths)
        
        return {
            "min_depth": min(all_depths),
            "max_depth": max(all_depths),
            "total_interval": max(all_depths) - min(all_depths)
        }
    
    def _calculate_petrophysical_properties(self, logs_data: Dict[LogType, WellLogData]) -> Dict[str, Any]:
        """Calculate petrophysical properties from logs."""
        properties = {}
        
        # Porosity calculation
        if LogType.NEUTRON in logs_data and LogType.DENSITY in logs_data:
            porosity_results = self._calculate_porosity(
                logs_data[LogType.NEUTRON], logs_data[LogType.DENSITY]
            )
            properties["porosity"] = porosity_results
        
        # Water saturation calculation (Archie's equation)
        if LogType.RESISTIVITY in logs_data and "porosity" in properties:
            sw_results = self._calculate_water_saturation(
                logs_data[LogType.RESISTIVITY], properties["porosity"]
            )
            properties["water_saturation"] = sw_results
        
        # Permeability estimation
        if "porosity" in properties:
            perm_results = self._estimate_permeability(properties["porosity"], logs_data)
            properties["permeability"] = perm_results
        
        return properties
    
    def _calculate_porosity(self, neutron_log: WellLogData, density_log: WellLogData) -> Dict[str, Any]:
        """Calculate porosity using neutron-density logs."""
        # Interpolate to common depth grid
        common_depths = np.linspace(
            max(min(neutron_log.depths), min(density_log.depths)),
            min(max(neutron_log.depths), max(density_log.depths)),
            min(len(neutron_log.depths), len(density_log.depths))
        )
        
        neutron_interp = np.interp(common_depths, neutron_log.depths, neutron_log.values)
        density_interp = np.interp(common_depths, density_log.depths, density_log.values)
        
        # Density porosity calculation
        rho_matrix = 2.65  # Sandstone matrix density
        rho_fluid = 1.0   # Water density
        
        porosity_density = (rho_matrix - density_interp) / (rho_matrix - rho_fluid)
        porosity_density = np.clip(porosity_density, 0, 0.5)  # Reasonable limits
        
        # Average neutron and density porosity
        porosity_avg = (neutron_interp + porosity_density) / 2
        
        return {
            "depths": common_depths.tolist(),
            "neutron_porosity": neutron_interp.tolist(),
            "density_porosity": porosity_density.tolist(),
            "average_porosity": porosity_avg.tolist(),
            "statistics": {
                "mean_porosity": float(np.mean(porosity_avg)),
                "std_porosity": float(np.std(porosity_avg)),
                "max_porosity": float(np.max(porosity_avg)),
                "min_porosity": float(np.min(porosity_avg))
            }
        }
    
    def _calculate_water_saturation(self, res_log: WellLogData, porosity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate water saturation using Archie's equation."""
        # Archie's parameters (typical values)
        a = 1.0      # Formation factor coefficient
        m = 2.0      # Cementation exponent
        n = 2.0      # Saturation exponent
        rw = 0.1     # Formation water resistivity (ohm-m)
        
        depths = np.array(porosity_data["depths"])
        porosity = np.array(porosity_data["average_porosity"])
        
        # Interpolate resistivity to porosity depths
        resistivity = np.interp(depths, res_log.depths, res_log.values)
        
        # Formation factor
        F = a / (porosity ** m)
        
        # Water saturation (Archie's equation)
        sw = (a * rw / (resistivity * (porosity ** m))) ** (1/n)
        sw = np.clip(sw, 0, 1)  # Limit between 0 and 1
        
        # Hydrocarbon saturation
        sh = 1 - sw
        
        return {
            "depths": depths.tolist(),
            "water_saturation": sw.tolist(),
            "hydrocarbon_saturation": sh.tolist(),
            "formation_factor": F.tolist(),
            "statistics": {
                "mean_sw": float(np.mean(sw)),
                "mean_sh": float(np.mean(sh)),
                "max_sh": float(np.max(sh)),
                "min_sw": float(np.min(sw))
            },
            "archie_parameters": {
                "a": a, "m": m, "n": n, "rw": rw
            }
        }
    
    def _estimate_permeability(self, porosity_data: Dict[str, Any], logs_data: Dict[LogType, WellLogData]) -> Dict[str, Any]:
        """Estimate permeability using empirical correlations."""
        porosity = np.array(porosity_data["average_porosity"])
        depths = np.array(porosity_data["depths"])
        
        # Kozeny-Carman type correlation
        # K = C * (phi^3) / (1-phi)^2
        C = 5000  # Empirical constant (depends on rock type)
        
        permeability = C * (porosity ** 3) / ((1 - porosity) ** 2)
        permeability = np.clip(permeability, 0.01, 10000)  # Reasonable limits (mD)
        
        return {
            "depths": depths.tolist(),
            "permeability_md": permeability.tolist(),
            "statistics": {
                "mean_perm": float(np.mean(permeability)),
                "max_perm": float(np.max(permeability)),
                "p90_perm": float(np.percentile(permeability, 90)),
                "p10_perm": float(np.percentile(permeability, 10))
            },
            "correlation_used": "kozeny_carman_modified"
        }
    
    def _detect_formation_boundaries(self, logs_data: Dict[LogType, WellLogData]) -> List[Dict[str, Any]]:
        """Detect formation boundaries using log signature changes."""
        boundaries = []
        
        if LogType.GAMMA_RAY in logs_data:
            gr_log = logs_data[LogType.GAMMA_RAY]
            gr_boundaries = self._detect_gr_boundaries(gr_log)
            boundaries.extend(gr_boundaries)
        
        if LogType.RESISTIVITY in logs_data:
            res_log = logs_data[LogType.RESISTIVITY]
            res_boundaries = self._detect_resistivity_boundaries(res_log)
            boundaries.extend(res_boundaries)
        
        # Consolidate boundaries
        consolidated = self._consolidate_boundaries(boundaries)
        
        return consolidated
    
    def _detect_gr_boundaries(self, gr_log: WellLogData) -> List[Dict[str, Any]]:
        """Detect formation boundaries using gamma ray log."""
        depths = np.array(gr_log.depths)
        values = np.array(gr_log.values)
        
        # Smooth the log
        smooth_values = signal.savgol_filter(values, 11, 2)
        
        # Calculate gradient
        gradient = np.gradient(smooth_values, depths)
        
        # Find significant gradient changes
        gradient_threshold = np.percentile(np.abs(gradient), 95)
        
        boundaries = []
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i]) > gradient_threshold:
                boundaries.append({
                    "depth": depths[i],
                    "method": "gamma_ray_gradient",
                    "gradient_value": gradient[i],
                    "gr_above": smooth_values[i-1],
                    "gr_below": smooth_values[i+1],
                    "boundary_type": "formation_boundary"
                })
        
        return boundaries
    
    def _detect_resistivity_boundaries(self, res_log: WellLogData) -> List[Dict[str, Any]]:
        """Detect boundaries using resistivity changes."""
        depths = np.array(res_log.depths)
        values = np.array(res_log.values)
        
        # Use log scale for resistivity
        log_res = np.log10(np.maximum(values, 0.1))
        
        # Smooth and calculate gradient
        smooth_log_res = signal.savgol_filter(log_res, 11, 2)
        gradient = np.gradient(smooth_log_res, depths)
        
        # Find significant changes
        threshold = np.percentile(np.abs(gradient), 90)
        
        boundaries = []
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i]) > threshold:
                boundaries.append({
                    "depth": depths[i],
                    "method": "resistivity_gradient",
                    "gradient_value": gradient[i],
                    "res_above": values[i-1],
                    "res_below": values[i+1],
                    "boundary_type": "formation_boundary"
                })
        
        return boundaries
    
    def _consolidate_boundaries(self, boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate nearby formation boundaries."""
        if not boundaries:
            return []
        
        # Sort by depth
        boundaries.sort(key=lambda x: x["depth"])
        
        # Merge boundaries within 10 feet
        consolidated = []
        merge_distance = 10.0
        
        i = 0
        while i < len(boundaries):
            current_boundary = boundaries[i].copy()
            methods_used = [current_boundary["method"]]
            
            # Look for nearby boundaries
            j = i + 1
            while j < len(boundaries) and boundaries[j]["depth"] - current_boundary["depth"] <= merge_distance:
                methods_used.append(boundaries[j]["method"])
                j += 1
            
            current_boundary["methods_used"] = methods_used
            current_boundary["confidence"] = len(methods_used) / 2.0  # More methods = higher confidence
            
            consolidated.append(current_boundary)
            i = j
        
        return consolidated
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate interpretation and drilling recommendations."""
        recommendations = []
        
        # Check log quality
        poor_quality_logs = []
        for log_type, quality in analysis_results.get("log_quality", {}).items():
            if quality.get("overall_quality") in ["poor", "unusable"]:
                poor_quality_logs.append(log_type)
        
        if poor_quality_logs:
            recommendations.append(
                f"Re-run or reprocess the following logs due to poor quality: {', '.join(poor_quality_logs)}"
            )
        
        # Check for potential hydrocarbon zones
        petrophysics = analysis_results.get("petrophysical_properties", {})
        if "water_saturation" in petrophysics:
            sw_stats = petrophysics["water_saturation"]["statistics"]
            if sw_stats.get("mean_sw", 1.0) < 0.6:
                recommendations.append("Potential hydrocarbon-bearing intervals identified based on water saturation")
        
        # Check porosity
        if "porosity" in petrophysics:
            por_stats = petrophysics["porosity"]["statistics"]
            if por_stats.get("mean_porosity", 0.0) > 0.15:
                recommendations.append("Good reservoir quality indicated by porosity values")
            elif por_stats.get("mean_porosity", 0.0) < 0.05:
                recommendations.append("Tight formation - consider stimulation for production")
        
        # Fluid contact recommendations
        contacts = analysis_results.get("fluid_contacts", [])
        if contacts:
            recommendations.append(
                f"Potential fluid contacts detected at depths: {[c.get('depth', 0) for c in contacts[:3]]}"
            )
        
        # Formation boundaries
        boundaries = analysis_results.get("formation_boundaries", [])
        if len(boundaries) > 5:
            recommendations.append("Multiple formation boundaries detected - consider detailed stratigraphic correlation")
        
        return recommendations


class LogQualityAnalyzer:
    """Analyze and assess well log data quality."""
    
    def assess_quality(self, log_data: WellLogData) -> Dict[str, Any]:
        """Comprehensive log quality assessment."""
        
        values = np.array(log_data.values)
        depths = np.array(log_data.depths)
        
        quality_metrics = {
            "completeness": self._assess_completeness(values),
            "consistency": self._assess_consistency(values),
            "noise_level": self._assess_noise(values),
            "outliers": self._detect_outliers(values),
            "depth_sampling": self._assess_depth_sampling(depths),
            "overall_quality": LogQuality.GOOD
        }
        
        # Determine overall quality
        quality_score = (
            quality_metrics["completeness"] * 0.3 +
            quality_metrics["consistency"] * 0.25 +
            (1 - quality_metrics["noise_level"]) * 0.25 +
            (1 - quality_metrics["outliers"]["outlier_fraction"]) * 0.2
        )
        
        if quality_score > 0.8:
            quality_metrics["overall_quality"] = LogQuality.EXCELLENT
        elif quality_score > 0.6:
            quality_metrics["overall_quality"] = LogQuality.GOOD
        elif quality_score > 0.4:
            quality_metrics["overall_quality"] = LogQuality.FAIR
        elif quality_score > 0.2:
            quality_metrics["overall_quality"] = LogQuality.POOR
        else:
            quality_metrics["overall_quality"] = LogQuality.UNUSABLE
        
        quality_metrics["quality_score"] = quality_score
        
        return quality_metrics
    
    def _assess_completeness(self, values: np.ndarray) -> float:
        """Assess data completeness (fraction of non-null values)."""
        valid_values = ~np.isnan(values) & ~np.isinf(values)
        return np.sum(valid_values) / len(values)
    
    def _assess_consistency(self, values: np.ndarray) -> float:
        """Assess data consistency (low variability in similar intervals)."""
        # Use moving standard deviation as consistency metric
        window_size = min(21, len(values) // 10)
        if window_size < 3:
            return 0.5
        
        moving_std = []
        for i in range(window_size, len(values) - window_size):
            window_std = np.std(values[i-window_size//2:i+window_size//2])
            moving_std.append(window_std)
        
        if not moving_std:
            return 0.5
        
        # Normalize consistency metric
        overall_std = np.std(values)
        avg_moving_std = np.mean(moving_std)
        
        if overall_std == 0:
            return 1.0
        
        consistency = 1 - min(avg_moving_std / overall_std, 1.0)
        return max(consistency, 0.0)
    
    def _assess_noise(self, values: np.ndarray) -> float:
        """Assess noise level in the data."""
        if len(values) < 10:
            return 0.5
        
        # High-frequency noise assessment
        diff = np.diff(values)
        second_diff = np.diff(diff)
        
        # Noise metric based on second derivative
        noise_metric = np.std(second_diff) / (np.std(values) + 1e-10)
        
        # Normalize to 0-1 range
        normalized_noise = min(noise_metric / 10.0, 1.0)
        
        return normalized_noise
    
    def _detect_outliers(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using statistical methods."""
        
        # Z-score method
        z_scores = np.abs(zscore(values))
        z_outliers = np.sum(z_scores > 3)
        
        # IQR method
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = np.sum((values < lower_bound) | (values > upper_bound))
        
        outlier_fraction = max(z_outliers, iqr_outliers) / len(values)
        
        return {
            "z_score_outliers": int(z_outliers),
            "iqr_outliers": int(iqr_outliers),
            "outlier_fraction": outlier_fraction,
            "outlier_indices": np.where((z_scores > 3) | (values < lower_bound) | (values > upper_bound))[0].tolist()
        }
    
    def _assess_depth_sampling(self, depths: np.ndarray) -> Dict[str, Any]:
        """Assess depth sampling quality."""
        
        depth_intervals = np.diff(depths)
        
        sampling_metrics = {
            "mean_interval": float(np.mean(depth_intervals)),
            "std_interval": float(np.std(depth_intervals)),
            "min_interval": float(np.min(depth_intervals)),
            "max_interval": float(np.max(depth_intervals)),
            "sampling_regularity": 1 - min(np.std(depth_intervals) / np.mean(depth_intervals), 1.0)
        }
        
        return sampling_metrics


# Global instances
well_log_ai = LogAnalysisEngine()
lithology_classifier = LithologyClassifier()
fluid_contact_detector = FluidContactDetector()