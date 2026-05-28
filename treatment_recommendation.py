# treatment_recommendation.py
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
from flask import render_template, Request , jsonify, request
import numpy as np
from datetime import datetime, timedelta

class TreatmentType(Enum):
    """Types of treatment actions"""
    LIFESTYLE = "lifestyle"
    MEDICATION = "medication"
    MONITORING = "monitoring"
    EDUCATIONAL = "educational"

class RiskLevel(Enum):
    """Patient risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PatientProfile:
    """Patient data structure"""
    age: int
    bmi: float
    glucose_level: float
    hba1c: float
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    insulin_level: float
    genetic_risk_score: float  # 0-1, higher means more genetic predisposition
    comorbidities: List[str]  # e.g., ['hypertension', 'obesity']
    preferences: Dict[str, bool]  # e.g., {'prefers_pills': True, 'prefers_exercise': False}
    budget_constraint: float  # monthly budget in USD
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score 0-100"""
        weights = {
            'bmi': 0.15,
            'glucose': 0.25,
            'hba1c': 0.20,
            'bp': 0.15,
            'age': 0.10,
            'genetic': 0.15
        }
        
        # Normalize values
        bmi_score = max(0, min(1, (self.bmi - 18.5) / (40 - 18.5)))
        glucose_score = max(0, min(1, (self.glucose_level - 70) / (200 - 70)))
        hba1c_score = max(0, min(1, (self.hba1c - 4.0) / (10.0 - 4.0)))
        bp_score = max(0, min(1, (self.blood_pressure_systolic - 110) / (180 - 110)))
        age_score = max(0, min(1, (self.age - 20) / (80 - 20)))
        
        risk_score = (
            weights['bmi'] * bmi_score +
            weights['glucose'] * glucose_score +
            weights['hba1c'] * hba1c_score +
            weights['bp'] * bp_score +
            weights['age'] * age_score +
            weights['genetic'] * self.genetic_risk_score
        )
        
        # Adjust for comorbidities
        comorbidity_multiplier = 1.0
        if 'hypertension' in self.comorbidities:
            comorbidity_multiplier *= 1.2
        if 'obesity' in self.comorbidities:
            comorbidity_multiplier *= 1.3
        if 'kidney_disease' in self.comorbidities:
            comorbidity_multiplier *= 1.4
        
        return min(100, risk_score * 100 * comorbidity_multiplier)
    
    def get_risk_level(self) -> RiskLevel:
        """Convert risk score to risk level"""
        score = self.calculate_risk_score()
        if score < 30:
            return RiskLevel.LOW
        elif score < 60:
            return RiskLevel.MEDIUM
        elif score < 85:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

@dataclass
class TreatmentAction:
    """Individual treatment action"""
    action_id: str
    action_type: TreatmentType
    description: str
    duration_days: int  # How long this action should be followed
    cost_per_month: float  # USD per month
    effectiveness_score: float  # 0-1, estimated effectiveness
    risk_reduction: float  # 0-1, estimated risk reduction
    side_effects_score: float  # 0-1, higher means more side effects
    constraints: Dict[str, any]  # Patient constraints this action satisfies
    prerequisites: List[str]  # action_ids that must be completed before this
    next_actions: List[str]  # Possible follow-up actions
    
    def calculate_utility(self, patient: PatientProfile) -> float:
        """Calculate utility score for this action for given patient"""
        base_score = self.effectiveness_score * 0.4 + \
                    (1 - self.side_effects_score) * 0.3 + \
                    self.risk_reduction * 0.3
        
        # Adjust for patient preferences
        preference_bonus = 0.0
        
        if self.action_type == TreatmentType.LIFESTYLE:
            if patient.preferences.get('prefers_exercise', False):
                preference_bonus += 0.2
            if patient.preferences.get('prefers_diet', False):
                preference_bonus += 0.2
        elif self.action_type == TreatmentType.MEDICATION:
            if patient.preferences.get('prefers_pills', True):
                preference_bonus += 0.2
        
        # Adjust for budget
        budget_penalty = 0.0
        if self.cost_per_month > patient.budget_constraint:
            budget_penalty = (self.cost_per_month - patient.budget_constraint) / patient.budget_constraint
        
        return max(0, base_score + preference_bonus - budget_penalty)

class TreatmentOptimizer:
    """Main optimizer using A* search to find optimal treatment path"""
    
    def __init__(self):
        self.treatment_library = self._initialize_treatment_library()
    
    def _initialize_treatment_library(self) -> Dict[str, TreatmentAction]:
        """Initialize with standard diabetes treatment actions"""
        treatments = {}
        
        # Lifestyle interventions
        treatments["diet_low_carb"] = TreatmentAction(
            action_id="diet_low_carb",
            action_type=TreatmentType.LIFESTYLE,
            description="Low carbohydrate diet (<50g carbs/day)",
            duration_days=30,
            cost_per_month=50.0,  # Cost of special foods
            effectiveness_score=0.7,
            risk_reduction=0.4,
            side_effects_score=0.1,
            constraints={"requires_motivation": True},
            prerequisites=[],
            next_actions=["diet_maintenance", "exercise_plan"]
        )
        
        treatments["exercise_moderate"] = TreatmentAction(
            action_id="exercise_moderate",
            action_type=TreatmentType.LIFESTYLE,
            description="Moderate exercise 30 min/day, 5 days/week",
            duration_days=30,
            cost_per_month=20.0,  # Gym membership or equipment
            effectiveness_score=0.6,
            risk_reduction=0.3,
            side_effects_score=0.05,
            constraints={"requires_physical_ability": True},
            prerequisites=[],
            next_actions=["exercise_intense", "monitor_glucose_frequent"]
        )
        
        treatments["exercise_intense"] = TreatmentAction(
            action_id="exercise_intense",
            action_type=TreatmentType.LIFESTYLE,
            description="Intense exercise 45 min/day, 5 days/week",
            duration_days=30,
            cost_per_month=30.0,
            effectiveness_score=0.8,
            risk_reduction=0.5,
            side_effects_score=0.15,
            constraints={"requires_good_fitness": True},
            prerequisites=["exercise_moderate"],
            next_actions=["maintenance_exercise"]
        )
        
        # Medications
        treatments["metformin_500mg"] = TreatmentAction(
            action_id="metformin_500mg",
            action_type=TreatmentType.MEDICATION,
            description="Metformin 500mg twice daily",
            duration_days=90,
            cost_per_month=15.0,
            effectiveness_score=0.7,
            risk_reduction=0.4,
            side_effects_score=0.25,  # GI side effects
            constraints={"requires_prescription": True},
            prerequisites=[],
            next_actions=["metformin_1000mg", "add_sulfonylurea"]
        )
        
        treatments["metformin_1000mg"] = TreatmentAction(
            action_id="metformin_1000mg",
            action_type=TreatmentType.MEDICATION,
            description="Metformin 1000mg twice daily",
            duration_days=90,
            cost_per_month=20.0,
            effectiveness_score=0.8,
            risk_reduction=0.5,
            side_effects_score=0.35,
            constraints={"requires_prescription": True},
            prerequisites=["metformin_500mg"],
            next_actions=["add_glp1", "add_insulin_basal"]
        )
        
        treatments["add_glp1"] = TreatmentAction(
            action_id="add_glp1",
            action_type=TreatmentType.MEDICATION,
            description="Add GLP-1 receptor agonist",
            duration_days=90,
            cost_per_month=300.0,
            effectiveness_score=0.9,
            risk_reduction=0.7,
            side_effects_score=0.3,
            constraints={"requires_prescription": True, "high_cost": True},
            prerequisites=["metformin_1000mg"],
            next_actions=["maintenance_glp1"]
        )
        
        treatments["insulin_basal"] = TreatmentAction(
            action_id="insulin_basal",
            action_type=TreatmentType.MEDICATION,
            description="Basal insulin once daily",
            duration_days=90,
            cost_per_month=100.0,
            effectiveness_score=0.95,
            risk_reduction=0.8,
            side_effects_score=0.4,  # Risk of hypoglycemia
            constraints={"requires_injection": True, "requires_monitoring": True},
            prerequisites=["metformin_1000mg"],
            next_actions=["insulin_basal_bolus"]
        )
        
        # Monitoring actions
        treatments["monitor_glucose_weekly"] = TreatmentAction(
            action_id="monitor_glucose_weekly",
            action_type=TreatmentType.MONITORING,
            description="Weekly glucose monitoring",
            duration_days=30,
            cost_per_month=40.0,  # Test strips
            effectiveness_score=0.3,
            risk_reduction=0.2,
            side_effects_score=0.0,
            constraints={"requires_glucometer": True},
            prerequisites=[],
            next_actions=["monitor_glucose_daily", "diet_low_carb"]
        )
        
        treatments["monitor_glucose_daily"] = TreatmentAction(
            action_id="monitor_glucose_daily",
            action_type=TreatmentType.MONITORING,
            description="Daily glucose monitoring",
            duration_days=30,
            cost_per_month=120.0,
            effectiveness_score=0.6,
            risk_reduction=0.4,
            side_effects_score=0.0,
            constraints={"requires_glucometer": True},
            prerequisites=["monitor_glucose_weekly"],
            next_actions=[]
        )
        
        # Educational actions
        treatments["diabetes_education"] = TreatmentAction(
            action_id="diabetes_education",
            action_type=TreatmentType.EDUCATIONAL,
            description="Diabetes self-management education",
            duration_days=7,
            cost_per_month=100.0,  # One-time cost
            effectiveness_score=0.5,
            risk_reduction=0.3,
            side_effects_score=0.0,
            constraints={"requires_time_commitment": True},
            prerequisites=[],
            next_actions=["diet_low_carb", "exercise_moderate", "monitor_glucose_weekly"]
        )
        
        return treatments
    
    def a_star_search(self, patient: PatientProfile, max_depth: int = 5) -> Tuple[List[str], float]:
        """
        A* search to find optimal treatment path
        
        Heuristic: Estimated remaining risk reduction needed
        Cost: Combination of side effects, cost, and duration
        
        Returns: (optimal_path, total_utility)
        """
        # Define start state
        start_state = {
            'completed_actions': set(),
            'current_actions': set(),
            'total_utility': 0.0,
            'total_cost': 0.0,
            'estimated_risk_reduction': 0.0,
            'path': []
        }
        
        # Frontier: priority queue (min-heap based on f-score)
        frontier = []
        heapq.heappush(frontier, (0.0, 0, start_state))  # (f_score, counter, state)
        
        counter = 1  # For tie-breaking
        
        best_solution = None
        best_utility = -float('inf')
        
        while frontier:
            f_score, _, state = heapq.heappop(frontier)
            
            # Check if we've reached maximum depth
            if len(state['path']) >= max_depth:
                if state['total_utility'] > best_utility:
                    best_utility = state['total_utility']
                    best_solution = state
                continue
            
            # Get possible next actions
            possible_actions = self._get_possible_actions(state, patient)
            
            # If no more actions, evaluate this path
            if not possible_actions:
                if state['total_utility'] > best_utility:
                    best_utility = state['total_utility']
                    best_solution = state
                continue
            
            for action_id in possible_actions:
                action = self.treatment_library[action_id]
                
                # Create new state
                new_state = {
                    'completed_actions': state['completed_actions'].copy(),
                    'current_actions': state['current_actions'].copy(),
                    'total_utility': state['total_utility'],
                    'total_cost': state['total_cost'],
                    'estimated_risk_reduction': state['estimated_risk_reduction'],
                    'path': state['path'].copy()
                }
                
                # Add action to state
                new_state['current_actions'].add(action_id)
                new_state['path'].append(action_id)
                
                # Update metrics
                action_utility = action.calculate_utility(patient)
                new_state['total_utility'] += action_utility
                new_state['total_cost'] += action.cost_per_month
                new_state['estimated_risk_reduction'] += action.risk_reduction
                
                # Mark prerequisites as completed
                for prereq in action.prerequisites:
                    new_state['completed_actions'].add(prereq)
                
                # Calculate heuristic: remaining risk to reduce
                remaining_risk = max(0, patient.calculate_risk_score()/100 - new_state['estimated_risk_reduction'])
                
                # Calculate f_score = -utility + heuristic (we use negative utility for min-heap)
                f_score = -new_state['total_utility'] + remaining_risk
                
                heapq.heappush(frontier, (f_score, counter, new_state))
                counter += 1
        
        if best_solution:
            return best_solution['path'], best_utility
        return [], 0.0
    
    def _get_possible_actions(self, state: Dict, patient: PatientProfile) -> List[str]:
        """Get actions that can be taken from current state"""
        possible = []
        
        for action_id, action in self.treatment_library.items():
            # Skip if already in path
            if action_id in state['path']:
                continue
            
            # Check prerequisites
            if not all(prereq in state['completed_actions'] for prereq in action.prerequisites):
                continue
            
            # Check patient constraints
            if not self._satisfies_constraints(action, patient):
                continue
            
            possible.append(action_id)
        
        return possible
    
    def _satisfies_constraints(self, action: TreatmentAction, patient: PatientProfile) -> bool:
        """Check if action satisfies patient constraints"""
        # Budget constraint
        if action.cost_per_month > patient.budget_constraint * 1.5:  # Allow 50% flexibility
            return False
        
        # Age constraints for certain actions
        if action.action_id == "exercise_intense" and patient.age > 70:
            return False
        
        # Comorbidity constraints
        if "insulin" in action.action_id and "kidney_disease" in patient.comorbidities:
            return False  # Insulin may need adjustment for kidney disease
        
        return True
    
    def generate_treatment_plan(self, patient: PatientProfile) -> Dict:
        """
        Generate complete treatment plan for patient
        
        Returns structured plan with timeline, actions, and metrics
        """
        # Get optimal path using A* search
        optimal_path, total_utility = self.a_star_search(patient)
        
        if not optimal_path:
            return self._get_fallback_plan(patient)
        
        # Build detailed plan
        plan = {
            "patient_id": id(patient),  # In real system, use actual patient ID
            "risk_level": patient.get_risk_level().value,
            "risk_score": patient.calculate_risk_score(),
            "generated_date": datetime.now().isoformat(),
            "total_estimated_utility": total_utility,
            "plan_duration_days": 0,
            "total_estimated_cost": 0.0,
            "total_estimated_risk_reduction": 0.0,
            "timeline": [],
            "follow_up_schedule": [],
            "constraints_satisfied": self._check_constraints_satisfaction(patient, optimal_path)
        }
        
        current_date = datetime.now()
        plan_duration = 0
        
        for action_id in optimal_path:
            action = self.treatment_library[action_id]
            
            # Calculate dates for this action
            start_date = current_date + timedelta(days=plan_duration)
            end_date = start_date + timedelta(days=action.duration_days)
            
            action_details = {
                "action_id": action_id,
                "type": action.action_type.value,
                "description": action.description,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": action.duration_days,
                "cost_per_month": action.cost_per_month,
                "total_cost": action.cost_per_month * (action.duration_days / 30),
                "effectiveness_score": action.effectiveness_score,
                "risk_reduction": action.risk_reduction,
                "utility_score": action.calculate_utility(patient),
                "instructions": self._get_action_instructions(action_id, patient)
            }
            
            plan["timeline"].append(action_details)
            plan["total_estimated_cost"] += action_details["total_cost"]
            plan["total_estimated_risk_reduction"] += action.risk_reduction
            plan_duration += action.duration_days
        
        plan["plan_duration_days"] = plan_duration
        
        # Generate follow-up schedule
        plan["follow_up_schedule"] = self._generate_follow_up_schedule(optimal_path, patient)
        
        return plan
    
    def _get_action_instructions(self, action_id: str, patient: PatientProfile) -> Dict:
        """Get detailed instructions for each action"""
        instructions = {
            "diet_low_carb": {
                "meals_per_day": "3 main meals + 2 snacks",
                "carb_limit": "50g per day",
                "foods_to_eat": ["Lean proteins", "Non-starchy vegetables", "Healthy fats"],
                "foods_to_avoid": ["Sugar", "White bread", "Pasta", "Potatoes"],
                "sample_meal_plan": "Breakfast: Eggs and avocado; Lunch: Grilled chicken salad; Dinner: Fish with broccoli"
            },
            "exercise_moderate": {
                "frequency": "5 days per week",
                "duration": "30 minutes per session",
                "types": ["Brisk walking", "Cycling", "Swimming"],
                "target_heart_rate": f"{int(0.5 * (220 - patient.age))}-{int(0.7 * (220 - patient.age))} bpm",
                "precautions": "Check glucose before and after exercise"
            },
            "metformin_500mg": {
                "dosage": "500mg twice daily",
                "timing": "With breakfast and dinner",
                "side_effects_watch": ["Nausea", "Diarrhea", "Stomach pain"],
                "contraindications": "Avoid if eGFR < 30",
                "monitoring": "Renal function tests every 6-12 months"
            },
            "monitor_glucose_weekly": {
                "frequency": "3 times per week",
                "times": ["Fasting", "2 hours after lunch", "Bedtime"],
                "target_ranges": {
                    "fasting": "80-130 mg/dL",
                    "post_prandial": "<180 mg/dL"
                },
                "log_format": "Date, Time, Value, Notes"
            }
        }
        
        return instructions.get(action_id, {"general": "Follow healthcare provider instructions"})
    
    def _generate_follow_up_schedule(self, path: List[str], patient: PatientProfile) -> List[Dict]:
        """Generate follow-up appointments based on treatment path"""
        schedule = []
        current_date = datetime.now()
        
        # Initial follow-up after first month
        schedule.append({
            "type": "initial_followup",
            "purpose": "Assess treatment response and side effects",
            "recommended_date": (current_date + timedelta(days=30)).isoformat(),
            "tests_recommended": ["HbA1c", "Fasting glucose", "Renal function"],
            "provider": "Primary care physician"
        })
        
        # Add follow-ups based on actions
        if any("insulin" in action_id for action_id in path):
            schedule.append({
                "type": "diabetes_specialist",
                "purpose": "Insulin therapy management",
                "recommended_date": (current_date + timedelta(days=60)).isoformat(),
                "tests_recommended": ["C-peptide", "Antibody panel"],
                "provider": "Endocrinologist"
            })
        
        if patient.get_risk_level() in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            schedule.append({
                "type": "comprehensive_assessment",
                "purpose": "Full diabetes complication screening",
                "recommended_date": (current_date + timedelta(days=90)).isoformat(),
                "tests_recommended": ["Retinal exam", "Foot exam", "Microalbuminuria", "Lipid profile"],
                "provider": "Diabetes care team"
            })
        
        # Regular quarterly follow-up
        for i in range(1, 4):
            schedule.append({
                "type": "routine_followup",
                "purpose": "Regular diabetes management check",
                "recommended_date": (current_date + timedelta(days=90 * i)).isoformat(),
                "tests_recommended": ["HbA1c", "Blood pressure", "Weight check"],
                "provider": "Primary care physician"
            })
        
        return schedule
    
    def _check_constraints_satisfaction(self, patient: PatientProfile, path: List[str]) -> Dict:
        """Check which constraints are satisfied by the plan"""
        constraints = {
            "budget": False,
            "patient_preferences": False,
            "safety": True,  # Assume safe unless proven otherwise
            "effectiveness": False
        }
        
        total_cost = sum(self.treatment_library[action_id].cost_per_month for action_id in path)
        constraints["budget"] = total_cost <= patient.budget_constraint * 1.2
        
        # Check preferences
        if patient.preferences.get('prefers_pills', False):
            constraints["patient_preferences"] = any(
                self.treatment_library[action_id].action_type == TreatmentType.MEDICATION 
                for action_id in path
            )
        else:
            constraints["patient_preferences"] = any(
                self.treatment_library[action_id].action_type == TreatmentType.LIFESTYLE 
                for action_id in path
            )
        
        # Check effectiveness (at least 0.5 total risk reduction)
        total_risk_reduction = sum(self.treatment_library[action_id].risk_reduction for action_id in path)
        constraints["effectiveness"] = total_risk_reduction >= 0.5
        
        return constraints
    
    def _get_fallback_plan(self, patient: PatientProfile) -> Dict:
        """Generate a fallback plan if optimization fails"""
        risk_level = patient.get_risk_level()
        
        if risk_level == RiskLevel.LOW:
            path = ["diabetes_education", "diet_low_carb", "exercise_moderate", "monitor_glucose_weekly"]
        elif risk_level == RiskLevel.MEDIUM:
            path = ["diabetes_education", "metformin_500mg", "diet_low_carb", "exercise_moderate", "monitor_glucose_weekly"]
        elif risk_level == RiskLevel.HIGH:
            path = ["diabetes_education", "metformin_1000mg", "diet_low_carb", "exercise_moderate", "monitor_glucose_daily"]
        else:  # CRITICAL
            path = ["diabetes_education", "metformin_1000mg", "add_glp1", "diet_low_carb", "exercise_moderate", "monitor_glucose_daily"]
        
        return self.generate_treatment_plan_from_path(patient, path)
    
    def generate_treatment_plan_from_path(self, patient: PatientProfile, path: List[str]) -> Dict:
        """Generate plan from a predefined path (for fallback)"""
        # Reuse the main generation logic but with predefined path
        temp_optimizer = TreatmentOptimizer()
        # We need to temporarily set the path
        # This is a simplified version
        plan = {
            "patient_id": id(patient),
            "risk_level": patient.get_risk_level().value,
            "risk_score": patient.calculate_risk_score(),
            "generated_date": datetime.now().isoformat(),
            "plan_duration_days": 0,
            "total_estimated_cost": 0.0,
            "total_estimated_risk_reduction": 0.0,
            "timeline": [],
            "follow_up_schedule": [],
            "note": "Fallback plan used"
        }
        
        return plan

# Flask integration module
class TreatmentAPI:
    """Flask API endpoints for treatment recommendations"""
    
    def __init__(self):
        self.optimizer = TreatmentOptimizer()
    
    def register_routes(self, app):
        """Register Flask routes"""
        
        @app.route('/api/treatment/recommend', methods=['POST'])
        def recommend_treatment():
            """Generate treatment plan based on patient data"""
            try:
                data = request.json
                
                # Validate required fields
                required_fields = ['age', 'bmi', 'glucose_level', 'hba1c', 
                                  'blood_pressure_systolic', 'blood_pressure_diastolic']
                missing = [f for f in required_fields if f not in data]
                if missing:
                    return jsonify({
                        "error": f"Missing required fields: {missing}",
                        "status": "error"
                    }), 400
                
                # Create patient profile
                patient = PatientProfile(
                    age=int(data['age']),
                    bmi=float(data['bmi']),
                    glucose_level=float(data['glucose_level']),
                    hba1c=float(data.get('hba1c', 6.5)),
                    blood_pressure_systolic=int(data['blood_pressure_systolic']),
                    blood_pressure_diastolic=int(data['blood_pressure_diastolic']),
                    insulin_level=float(data.get('insulin_level', 10.0)),
                    genetic_risk_score=float(data.get('genetic_risk_score', 0.3)),
                    comorbidities=data.get('comorbidities', []),
                    preferences=data.get('preferences', {}),
                    budget_constraint=float(data.get('budget_constraint', 200.0))
                )
                
                # Generate treatment plan
                plan = self.optimizer.generate_treatment_plan(patient)
                
                # Add algorithm explanation
                plan["algorithm_explanation"] = {
                    "method": "A* Search Algorithm",
                    "optimization_goal": "Maximize utility while minimizing risk",
                    "factors_considered": [
                        "Patient risk score",
                        "Treatment effectiveness",
                        "Side effects",
                        "Cost constraints",
                        "Patient preferences",
                        "Medical constraints"
                    ],
                    "search_depth": 5,
                    "states_explored": "Dynamically based on constraints"
                }
                
                return jsonify({
                    "status": "success",
                    "treatment_plan": plan,
                    "patient_summary": {
                        "risk_level": patient.get_risk_level().value,
                        "risk_score": patient.calculate_risk_score(),
                        "age": patient.age,
                        "bmi_category": self._get_bmi_category(patient.bmi)
                    }
                })
                
            except Exception as e:
                app.logger.error(f"Treatment recommendation error: {str(e)}")
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500
        
        @app.route('/api/treatment/options', methods=['GET'])
        def get_treatment_options():
            """Get all available treatment options"""
            options = {}
            for action_id, action in self.optimizer.treatment_library.items():
                options[action_id] = {
                    "type": action.action_type.value,
                    "description": action.description,
                    "cost_per_month": action.cost_per_month,
                    "effectiveness": action.effectiveness_score,
                    "risk_reduction": action.risk_reduction,
                    "side_effects": action.side_effects_score,
                    "duration_days": action.duration_days
                }
            
            return jsonify({
                "status": "success",
                "total_options": len(options),
                "options": options
            })
        
        @app.route('/api/treatment/simulate', methods=['POST'])
        def simulate_treatment():
            """Simulate treatment outcome for a specific plan"""
            try:
                data = request.json
                patient_data = data.get('patient_data', {})
                treatment_plan = data.get('treatment_plan', [])
                
                # Create patient profile
                patient = PatientProfile(
                    age=int(patient_data['age']),
                    bmi=float(patient_data['bmi']),
                    glucose_level=float(patient_data['glucose_level']),
                    hba1c=float(patient_data.get('hba1c', 6.5)),
                    blood_pressure_systolic=int(patient_data['blood_pressure_systolic']),
                    blood_pressure_diastolic=int(patient_data['blood_pressure_diastolic']),
                    insulin_level=float(patient_data.get('insulin_level', 10.0)),
                    genetic_risk_score=float(patient_data.get('genetic_risk_score', 0.3)),
                    comorbidities=patient_data.get('comorbidities', []),
                    preferences=patient_data.get('preferences', {}),
                    budget_constraint=float(patient_data.get('budget_constraint', 200.0))
                )
                
                # Calculate projected outcomes
                total_cost = 0.0
                total_risk_reduction = 0.0
                total_utility = 0.0
                
                for action_id in treatment_plan:
                    if action_id in self.optimizer.treatment_library:
                        action = self.optimizer.treatment_library[action_id]
                        total_cost += action.cost_per_month
                        total_risk_reduction += action.risk_reduction
                        total_utility += action.calculate_utility(patient)
                
                # Project glucose reduction (simplified model)
                initial_glucose = patient.glucose_level
                projected_glucose = initial_glucose * (1 - total_risk_reduction * 0.7)
                
                return jsonify({
                    "status": "success",
                    "simulation_results": {
                        "initial_risk_score": patient.calculate_risk_score(),
                        "projected_risk_score": patient.calculate_risk_score() * (1 - total_risk_reduction),
                        "initial_glucose": initial_glucose,
                        "projected_glucose": projected_glucose,
                        "total_cost_3_months": total_cost * 3,
                        "total_risk_reduction": total_risk_reduction,
                        "total_utility": total_utility,
                        "estimated_hba1c_reduction": total_risk_reduction * 3  # Simplified
                    },
                    "recommendations": self._get_simulation_recommendations(total_utility, total_risk_reduction)
                })
                
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Categorize BMI"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def _get_simulation_recommendations(self, utility: float, risk_reduction: float) -> List[str]:
        """Get recommendations based on simulation results"""
        recommendations = []
        
        if utility < 0.5:
            recommendations.append("Consider adjusting treatment plan for better patient preference alignment")
        
        if risk_reduction < 0.3:
            recommendations.append("More aggressive treatment may be needed for adequate risk reduction")
        
        if risk_reduction > 0.8:
            recommendations.append("Excellent projected risk reduction - monitor for side effects")
        
        return recommendations

# Integration with existing Flask app
def integrate_treatment_module(app):
    """Integrate treatment recommendation module with existing Flask app"""
    treatment_api = TreatmentAPI()
    treatment_api.register_routes(app)
    
    # Add to navigation
    @app.context_processor
    def inject_navigation():
        return dict(
            nav_items=[
                {"name": "Home", "url": "/", "icon": "fas fa-home"},
                {"name": "Diabetes Prediction", "url": "/predict", "icon": "fas fa-robot"},
                {"name": "Symptom Search", "url": "/symptom_search", "icon": "fas fa-search-medical"},
                {"name": "Treatment Planner", "url": "/treatment", "icon": "fas fa-pills"},
                {"name": "Dashboard", "url": "/dashboard", "icon": "fas fa-chart-line"},
                {"name": "About", "url": "/about", "icon": "fas fa-info-circle"}
            ]
        )
    
    # Add treatment planner page
    @app.route('/treatment')
    def treatment_planner():
        return render_template('treatment_planner.html')