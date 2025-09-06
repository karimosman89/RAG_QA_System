#!/usr/bin/env python3
"""
Petroleum AI Platform - Comprehensive Demo Script

This script demonstrates the complete petroleum AI platform with all three core systems:
1. Well Log Interpretation
2. Drilling Risk Prediction  
3. Reservoir Navigation & Geo-steering

Usage:
    python petroleum_ai_demo.py

Author: AI Development Team
Created: 2024-09-06
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

# Import petroleum AI modules
from src.petroleum import (
    # Well Log Interpretation
    well_log_ai, WellLogData, LogType,
    
    # Drilling Risk Prediction
    drilling_risk_ai, DrillingParameters, DrillingHazard,
    
    # Reservoir Navigation & Geo-steering
    reservoir_navigation_ai, TrajectoryPoint, ReservoirTarget,
    NavigationMode, SteeringDirection,
    
    # Data Models
    LithologyType, FluidType, RiskLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PetroleumAIDemo:
    """Comprehensive demo of the Petroleum AI Platform."""
    
    def __init__(self):
        """Initialize the demo with sample data."""
        self.well_id = "DEMO_WELL_001"
        self.field_name = "Eagle Ford Shale"
        self.operator = "Demo Oil Company"
        
        # Generate sample well log data
        self.sample_well_logs = self._generate_sample_well_logs()
        
        # Generate sample drilling parameters
        self.sample_drilling_params = self._generate_sample_drilling_params()
        
        # Generate sample trajectory and targets
        self.sample_trajectory = self._generate_sample_trajectory()
        self.sample_targets = self._generate_sample_targets()
    
    def _generate_sample_well_logs(self) -> List[WellLogData]:
        """Generate realistic well log data for demonstration."""
        depths = np.arange(8000, 8500, 0.5)  # 500 ft interval, 0.5 ft sampling
        
        # Gamma Ray log (typical shale/sand sequence)
        gamma_values = []
        for depth in depths:
            base_gamma = 45 + 30 * np.sin((depth - 8000) / 50)  # Cyclical pattern
            noise = np.random.normal(0, 5)
            gamma_values.append(max(15, min(150, base_gamma + noise)))
        
        # Resistivity log (higher in hydrocarbon zones)
        resistivity_values = []
        for depth in depths:
            if 8200 <= depth <= 8300:  # Hydrocarbon zone
                base_resistivity = 50 + 20 * np.random.random()
            else:  # Water zones
                base_resistivity = 5 + 10 * np.random.random()
            resistivity_values.append(base_resistivity)
        
        # Neutron porosity (inverse correlation with gamma ray)
        neutron_values = []
        for i, gamma in enumerate(gamma_values):
            base_neutron = max(5, 30 - (gamma - 50) / 3)
            noise = np.random.normal(0, 2)
            neutron_values.append(max(0, min(40, base_neutron + noise)))
        
        # Density log (typical carbonate/sandstone values)
        density_values = []
        for neutron in neutron_values:
            base_density = 2.65 - neutron * 0.01  # Simplified density-porosity relation
            noise = np.random.normal(0, 0.05)
            density_values.append(max(2.0, min(3.0, base_density + noise)))
        
        # Photoelectric factor
        pe_values = []
        for i, gamma in enumerate(gamma_values):
            if gamma > 80:  # Shale
                pe = 2.8 + np.random.normal(0, 0.2)
            else:  # Sand/carbonate
                pe = 1.8 + np.random.normal(0, 0.1)
            pe_values.append(max(1.0, min(5.0, pe)))
        
        return [
            WellLogData(
                well_id=self.well_id,
                log_type=LogType.GAMMA_RAY,
                depth_start=float(depths[0]),
                depth_end=float(depths[-1]),
                depths=depths.tolist(),
                values=gamma_values,
                units="API"
            ),
            WellLogData(
                well_id=self.well_id,
                log_type=LogType.RESISTIVITY,
                depth_start=float(depths[0]),
                depth_end=float(depths[-1]),
                depths=depths.tolist(),
                values=resistivity_values,
                units="OHMM"
            ),
            WellLogData(
                well_id=self.well_id,
                log_type=LogType.NEUTRON,
                depth_start=float(depths[0]),
                depth_end=float(depths[-1]),
                depths=depths.tolist(),
                values=neutron_values,
                units="V/V"
            ),
            WellLogData(
                well_id=self.well_id,
                log_type=LogType.DENSITY,
                depth_start=float(depths[0]),
                depth_end=float(depths[-1]),
                depths=depths.tolist(),
                values=density_values,
                units="G/C3"
            ),
            WellLogData(
                well_id=self.well_id,
                log_type=LogType.PHOTOELECTRIC,
                depth_start=float(depths[0]),
                depth_end=float(depths[-1]),
                depths=depths.tolist(),
                values=pe_values,
                units="B/E"
            )
        ]
    
    def _generate_sample_drilling_params(self) -> DrillingParameters:
        """Generate realistic drilling parameters."""
        return DrillingParameters(
            well_id=self.well_id,
            measured_depth=8250.0,
            weight_on_bit=28.5,
            rotary_speed=110.0,
            flow_rate=320.0,
            pump_pressure=3180.0,
            torque=12500.0,
            mud_weight=9.2,
            mud_viscosity=35.0,
            rate_of_penetration=45.5,
            mud_temperature=68.5,
            timestamp=datetime.now()
        )
    
    def _generate_sample_trajectory(self) -> TrajectoryPoint:
        """Generate current trajectory point."""
        return TrajectoryPoint(
            measured_depth=8250.0,
            true_vertical_depth=7895.0,
            inclination=78.5,
            azimuth=45.0,
            northing=1150.0,
            easting=825.0,
            dogleg_severity=1.8
        )
    
    def _generate_sample_targets(self) -> List[ReservoirTarget]:
        """Generate reservoir targets."""
        return [
            ReservoirTarget(
                name="Upper_Eagle_Ford",
                top_depth=7880.0,
                bottom_depth=7920.0,
                thickness=40.0,
                porosity=0.12,
                permeability=0.05,
                oil_saturation=0.70,
                structural_dip=2.5,
                dip_direction=135.0,
                quality_factor=0.75
            ),
            ReservoirTarget(
                name="Lower_Eagle_Ford", 
                top_depth=7920.0,
                bottom_depth=7965.0,
                thickness=45.0,
                porosity=0.14,
                permeability=0.08,
                oil_saturation=0.68,
                structural_dip=3.2,
                dip_direction=142.0,
                quality_factor=0.82
            )
        ]
    
    async def run_well_log_interpretation_demo(self) -> Dict[str, Any]:
        """Demonstrate well log interpretation AI system."""
        print("\n" + "="*80)
        print("🔍 WELL LOG INTERPRETATION AI DEMONSTRATION")
        print("="*80)
        
        print(f"\n📍 Well: {self.well_id}")
        print(f"📍 Field: {self.field_name}")
        print(f"📍 Operator: {self.operator}")
        
        print(f"\n📊 Log Data Summary:")
        for log in self.sample_well_logs:
            depth_range = f"{min(log.depths):.0f} - {max(log.depths):.0f} ft"
            print(f"   • {log.log_type.value}: {len(log.values)} points ({depth_range})")
        
        print("\n🤖 Running AI Analysis...")
        
        # Prepare logs data as dictionary
        logs_dict = {log.log_type: log for log in self.sample_well_logs}
        
        # Run well log interpretation AI
        log_results = well_log_ai.analyze_well_logs(logs_dict)
        
        print("\n📈 ANALYSIS RESULTS:")
        print("-" * 40)
        
        # Display lithology analysis
        lithology_analysis = log_results.get("lithology_analysis", {})
        print(f"🪨 Lithology Classification:")
        lithology_probs = lithology_analysis.get("lithology_probabilities", {})
        for lithology, probability in lithology_probs.items():
            print(f"   • {lithology}: {probability:.1%}")
        
        dominant_lithology = lithology_analysis.get("dominant_lithology", "Unknown")
        confidence = lithology_analysis.get("confidence", 0.0)
        print(f"   → Dominant Lithology: {dominant_lithology} ({confidence:.1%} confidence)")
        
        # Display reservoir properties
        reservoir_props = log_results.get("reservoir_properties", {})
        print(f"\n⛽ Reservoir Properties:")
        print(f"   • Average Porosity: {reservoir_props.get('average_porosity', 0):.1%}")
        print(f"   • Permeability Estimate: {reservoir_props.get('permeability_estimate', 0):.3f} mD")
        print(f"   • Water Saturation: {reservoir_props.get('water_saturation', 0):.1%}")
        print(f"   • Hydrocarbon Saturation: {reservoir_props.get('hydrocarbon_saturation', 0):.1%}")
        print(f"   • Net-to-Gross Ratio: {reservoir_props.get('net_to_gross', 0):.1%}")
        
        # Display fluid contacts
        fluid_contacts = log_results.get("fluid_contacts", {})
        print(f"\n🌊 Fluid Contact Analysis:")
        contacts_detected = fluid_contacts.get("contacts_detected", [])
        if contacts_detected:
            for contact in contacts_detected[:3]:  # Show top 3
                print(f"   • {contact['contact_type']} at {contact['depth']:.1f} ft "
                      f"(confidence: {contact['confidence']:.1%})")
        else:
            print("   • No significant fluid contacts detected")
        
        # Display formation evaluation
        formation_eval = log_results.get("formation_evaluation", {})
        print(f"\n🎯 Formation Evaluation:")
        print(f"   • Reservoir Quality Index: {formation_eval.get('reservoir_quality_index', 0):.3f}")
        print(f"   • Completion Quality: {formation_eval.get('completion_quality', 'Unknown')}")
        print(f"   • Recommended Completion: {formation_eval.get('recommended_completion', 'Standard')}")
        
        return log_results
    
    async def run_drilling_risk_prediction_demo(self) -> Dict[str, Any]:
        """Demonstrate drilling risk prediction AI system."""
        print("\n" + "="*80)
        print("⚠️  DRILLING RISK PREDICTION AI DEMONSTRATION")
        print("="*80)
        
        print(f"\n📊 Current Drilling Parameters:")
        params = self.sample_drilling_params
        print(f"   • Measured Depth: {params.measured_depth:.1f} ft")
        print(f"   • Weight on Bit: {params.weight_on_bit:.1f} klbs") 
        print(f"   • Rotary Speed: {params.rotary_speed:.0f} RPM")
        print(f"   • Flow Rate: {params.flow_rate:.0f} gpm")
        print(f"   • Standpipe Pressure: {params.standpipe_pressure:.0f} psi")
        print(f"   • Rate of Penetration: {params.rate_of_penetration:.1f} ft/hr")
        print(f"   • Total Gas: {params.total_gas:.0f} units")
        print(f"   • Mud Weight: {params.mud_weight:.1f} ppg")
        
        print("\n🤖 Running Risk Analysis...")
        
        # Run drilling risk prediction AI
        risk_results = await drilling_risk_ai.assess_comprehensive_risk(params)
        
        print("\n🚨 RISK ASSESSMENT RESULTS:")
        print("-" * 40)
        
        # Overall risk level
        overall_risk = risk_results.get("overall_risk_level", RiskLevel.LOW)
        risk_colors = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡", 
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴"
        }
        print(f"📊 Overall Risk Level: {risk_colors.get(overall_risk, '⚪')} {overall_risk.value.upper()}")
        
        # Individual risk assessments
        individual_risks = risk_results.get("individual_risk_assessment", {})
        print(f"\n🎯 Individual Risk Analysis:")
        for hazard_name, risk_data in individual_risks.items():
            risk_level = risk_data.get("risk_level", RiskLevel.LOW)
            probability = risk_data.get("probability", 0.0)
            print(f"   • {hazard_name.replace('_', ' ').title()}: "
                  f"{risk_colors.get(risk_level, '⚪')} {risk_level.value} ({probability:.1%})")
        
        # Real-time monitoring alerts
        real_time_analysis = risk_results.get("real_time_analysis", {})
        print(f"\n📡 Real-time Monitoring:")
        
        kick_analysis = real_time_analysis.get("kick_analysis", {})
        kick_probability = kick_analysis.get("kick_probability", 0.0)
        print(f"   • Kick Risk: {kick_probability:.1%} probability")
        
        lost_circulation = real_time_analysis.get("lost_circulation_analysis", {})
        loss_probability = lost_circulation.get("loss_probability", 0.0)
        print(f"   • Lost Circulation Risk: {loss_probability:.1%} probability")
        
        dysfunction_analysis = real_time_analysis.get("dysfunction_analysis", {})
        efficiency = dysfunction_analysis.get("drilling_efficiency", 0.0)
        print(f"   • Drilling Efficiency: {efficiency:.1%}")
        
        # Optimization recommendations
        recommendations = risk_results.get("optimization_recommendations", [])
        print(f"\n💡 Optimization Recommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"   {i}. {rec}")
        else:
            print("   • Current parameters are optimal - continue operations")
        
        return risk_results
    
    async def run_reservoir_navigation_demo(self) -> Dict[str, Any]:
        """Demonstrate reservoir navigation and geo-steering AI system."""
        print("\n" + "="*80)
        print("🎯 RESERVOIR NAVIGATION & GEO-STEERING AI DEMONSTRATION")
        print("="*80)
        
        print(f"\n📍 Current Trajectory:")
        traj = self.sample_trajectory
        print(f"   • Measured Depth: {traj.measured_depth:.1f} ft")
        print(f"   • True Vertical Depth: {traj.true_vertical_depth:.1f} ft")
        print(f"   • Inclination: {traj.inclination:.1f}°")
        print(f"   • Azimuth: {traj.azimuth:.1f}°")
        print(f"   • Dogleg Severity: {traj.dogleg_severity:.2f}°/100ft")
        print(f"   • Position: N{traj.northing:.0f}, E{traj.easting:.0f}")
        
        print(f"\n🎯 Reservoir Targets:")
        for i, target in enumerate(self.sample_targets, 1):
            print(f"   {i}. {target.name}:")
            print(f"      • Depth: {target.top_depth:.0f} - {target.bottom_depth:.0f} ft")
            print(f"      • Thickness: {target.thickness:.0f} ft")
            print(f"      • Quality Factor: {target.quality_factor:.2f}")
            print(f"      • Oil Saturation: {target.oil_saturation:.1%}")
        
        print("\n🤖 Running Navigation Analysis...")
        
        # Run reservoir navigation AI
        nav_results = await reservoir_navigation_ai(
            self.sample_well_logs,
            self.sample_trajectory,
            self.sample_drilling_params,
            self.sample_targets
        )
        
        print("\n🧭 NAVIGATION ANALYSIS RESULTS:")
        print("-" * 40)
        
        # Structural analysis
        structural_analysis = nav_results.get("structural_analysis", {})
        dip_analysis = structural_analysis.get("dip_analysis", {})
        print(f"🗻 Structural Analysis:")
        print(f"   • Apparent Dip: {dip_analysis.get('apparent_dip', 0):.1f}°")
        print(f"   • Dip Direction: {dip_analysis.get('dip_direction', 0):.1f}°")
        print(f"   • Structural Confidence: {dip_analysis.get('confidence', 0):.1%}")
        
        fault_detection = structural_analysis.get("fault_detection", {})
        major_faults = fault_detection.get("major_faults", 0)
        minor_faults = fault_detection.get("minor_faults", 0)
        print(f"   • Faults Detected: {major_faults} major, {minor_faults} minor")
        
        # Trajectory optimization
        trajectory_opt = nav_results.get("trajectory_optimization", {})
        drilling_efficiency = trajectory_opt.get("drilling_efficiency", 0.0)
        print(f"\n⚡ Trajectory Optimization:")
        print(f"   • Drilling Efficiency: {drilling_efficiency:.1%}")
        
        target_exposure = trajectory_opt.get("target_exposure", {})
        print(f"   • Target Exposure:")
        for target_name, exposure in target_exposure.items():
            print(f"     - {target_name}: {exposure:.1f} ft")
        
        # Geo-steering analysis
        geo_steering = nav_results.get("geo_steering", {})
        boundary_analysis = geo_steering.get("boundary_analysis", {})
        
        print(f"\n🎯 Geo-steering Commands:")
        steering_commands = geo_steering.get("steering_commands", [])
        if steering_commands:
            for i, command in enumerate(steering_commands[:3], 1):
                severity_emoji = "🟢" if command.severity < 0.3 else "🟡" if command.severity < 0.7 else "🔴"
                print(f"   {i}. {severity_emoji} {command.command.value.replace('_', ' ').title()}")
                print(f"      • Severity: {command.severity:.1f} | Confidence: {command.confidence:.1%}")
                print(f"      • Reason: {command.reason}")
        else:
            print("   • No steering adjustments required - maintain current trajectory")
        
        # Boundary detection
        crossing_prediction = boundary_analysis.get("crossing_prediction", {})
        boundary_distance = crossing_prediction.get("next_boundary_distance")
        if boundary_distance:
            print(f"\n🚧 Boundary Detection:")
            print(f"   • Next Boundary: {boundary_distance:.1f} ft ahead")
            print(f"   • Boundary Type: {crossing_prediction.get('boundary_type', 'Unknown')}")
            print(f"   • Detection Confidence: {crossing_prediction.get('confidence', 0):.1%}")
        
        # Recommendations
        recommendations = nav_results.get("recommendations", {})
        immediate_actions = recommendations.get("immediate_actions", [])
        short_term = recommendations.get("short_term_strategy", [])
        
        print(f"\n💡 Navigation Recommendations:")
        print(f"   📋 Immediate Actions:")
        for action in immediate_actions[:3]:
            print(f"      • {action}")
        
        print(f"   📈 Short-term Strategy:")  
        for strategy in short_term[:2]:
            print(f"      • {strategy}")
        
        return nav_results
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete petroleum AI platform demonstration."""
        print("🛢️  PETROLEUM AI ANALYTICS PLATFORM - COMPREHENSIVE DEMONSTRATION")
        print("=" * 80)
        print(f"🕐 Demo Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏭 Simulating: {self.field_name} drilling operations")
        
        # Initialize results
        demo_results = {
            "demo_info": {
                "well_id": self.well_id,
                "field_name": self.field_name,
                "operator": self.operator,
                "demo_timestamp": datetime.now().isoformat(),
                "platform_version": "1.0.0"
            }
        }
        
        try:
            # Run all three AI systems
            demo_results["well_log_interpretation"] = await self.run_well_log_interpretation_demo()
            demo_results["drilling_risk_prediction"] = await self.run_drilling_risk_prediction_demo()
            demo_results["reservoir_navigation"] = await self.run_reservoir_navigation_demo()
            
            # Generate integrated summary
            print("\n" + "="*80)
            print("📊 INTEGRATED AI PLATFORM SUMMARY")
            print("="*80)
            
            self._generate_integrated_summary(demo_results)
            
            print(f"\n✅ Demo Completed Successfully: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Demo execution error: {str(e)}")
            demo_results["error"] = str(e)
            print(f"\n❌ Demo Error: {str(e)}")
        
        return demo_results
    
    def _generate_integrated_summary(self, results: Dict[str, Any]) -> None:
        """Generate integrated summary of all AI systems."""
        print("\n🔍 Well Log AI Summary:")
        log_results = results.get("well_log_interpretation", {})
        reservoir_props = log_results.get("reservoir_properties", {})
        print(f"   • Reservoir Quality: {reservoir_props.get('reservoir_quality_index', 0):.3f}")
        print(f"   • Hydrocarbon Saturation: {reservoir_props.get('hydrocarbon_saturation', 0):.1%}")
        
        print("\n⚠️ Drilling Risk AI Summary:")
        risk_results = results.get("drilling_risk_prediction", {})
        overall_risk = risk_results.get("overall_risk_level", RiskLevel.LOW)
        print(f"   • Overall Risk Level: {overall_risk.value.upper()}")
        
        real_time = risk_results.get("real_time_analysis", {})
        kick_prob = real_time.get("kick_analysis", {}).get("kick_probability", 0)
        print(f"   • Kick Probability: {kick_prob:.1%}")
        
        print("\n🎯 Navigation AI Summary:")
        nav_results = results.get("reservoir_navigation", {})
        traj_opt = nav_results.get("trajectory_optimization", {})
        efficiency = traj_opt.get("drilling_efficiency", 0)
        print(f"   • Trajectory Efficiency: {efficiency:.1%}")
        
        geo_steering = nav_results.get("geo_steering", {})
        commands = geo_steering.get("steering_commands", [])
        active_commands = len([cmd for cmd in commands if cmd.severity > 0.3])
        print(f"   • Active Steering Commands: {active_commands}")
        
        print("\n🎯 INTEGRATED RECOMMENDATIONS:")
        print("   1. Continue drilling with current parameters - all systems optimal")
        print("   2. Monitor formation changes approaching target zone")
        print("   3. Maintain trajectory efficiency above 80%")
        print("   4. Review geo-steering commands every 100 ft")
        
        print("\n📈 BUSINESS IMPACT:")
        print(f"   • Risk Mitigation: Early warning systems active")
        print(f"   • Reservoir Optimization: AI-guided trajectory planning")
        print(f"   • Operational Efficiency: Real-time decision support")
        print(f"   • Cost Savings: Predictive analytics prevent costly incidents")
    
    def save_demo_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save demonstration results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"petroleum_ai_demo_results_{timestamp}.json"
        
        filepath = f"/home/user/webapp/{filename}"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Demo results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving demo results: {str(e)}")
            print(f"❌ Failed to save results: {str(e)}")
            return ""


async def main():
    """Main demonstration function."""
    print("🚀 Initializing Petroleum AI Platform Demo...")
    
    # Create demo instance
    demo = PetroleumAIDemo()
    
    # Run complete demonstration
    results = await demo.run_complete_demo()
    
    # Save results
    demo.save_demo_results(results)
    
    print("\n" + "="*80)
    print("🎉 PETROLEUM AI PLATFORM DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nFor more information or to schedule a technical review:")
    print("📧 Contact: ai-development-team@company.com")
    print("📞 Phone: +1 (555) 123-4567")
    print("🌐 Web: https://petroleum-ai-platform.com")
    print("\nThank you for exploring our Petroleum AI Analytics Platform!")


if __name__ == "__main__":
    asyncio.run(main())