# symptom_search.py
from collections import deque, defaultdict
import heapq
from typing import List, Dict, Tuple, Set
import json
from flask import render_template, Request , jsonify, request
class SymptomToDiseaseMapper:
    """Graph-based symptom to disease mapper using BFS/DFS algorithms"""
    
    def __init__(self):
        self.graph = self.build_medical_graph()
        self.disease_info = self.load_disease_info()
        self.symptom_descriptions = self.load_symptom_descriptions()
    
    def build_medical_graph(self) -> Dict[str, List[Tuple[str, float, str]]]:
        """Build adjacency list representation of medical knowledge"""
        graph = defaultdict(list)
        
        # Symptom → Disease connections with weights and relationship types
        # Diabetes-related symptoms
        graph["Frequent urination"].extend([
            ("Type 2 Diabetes", 0.9, "primary"),
            ("Type 1 Diabetes", 0.85, "primary"),
            ("Gestational Diabetes", 0.8, "primary"),
            ("Urinary Tract Infection", 0.4, "differential")
        ])
        
        graph["Excessive thirst"].extend([
            ("Type 2 Diabetes", 0.85, "primary"),
            ("Type 1 Diabetes", 0.9, "primary"),
            ("Dehydration", 0.6, "differential"),
            ("Diabetes Insipidus", 0.7, "differential")
        ])
        
        graph["Extreme hunger"].extend([
            ("Type 2 Diabetes", 0.8, "primary"),
            ("Type 1 Diabetes", 0.85, "primary"),
            ("Hyperthyroidism", 0.5, "differential")
        ])
        
        graph["Unexplained weight loss"].extend([
            ("Type 1 Diabetes", 0.9, "primary"),
            ("Type 2 Diabetes", 0.7, "secondary"),
            ("Cancer", 0.6, "differential"),
            ("Hyperthyroidism", 0.65, "differential")
        ])
        
        graph["Fatigue"].extend([
            ("Type 2 Diabetes", 0.6, "common"),
            ("Anemia", 0.7, "common"),
            ("Hypothyroidism", 0.65, "common"),
            ("Chronic Fatigue Syndrome", 0.8, "differential")
        ])
        
        graph["Blurred vision"].extend([
            ("Type 2 Diabetes", 0.7, "complication"),
            ("Diabetic Retinopathy", 0.9, "primary"),
            ("Cataracts", 0.6, "differential"),
            ("Glaucoma", 0.5, "differential")
        ])
        
        graph["Slow healing wounds"].extend([
            ("Type 2 Diabetes", 0.8, "complication"),
            ("Peripheral Artery Disease", 0.6, "differential"),
            ("Immunodeficiency", 0.5, "differential")
        ])
        
        graph["Frequent infections"].extend([
            ("Type 2 Diabetes", 0.7, "complication"),
            ("Immunodeficiency", 0.8, "differential")
        ])
        
        graph["Tingling/numbness in hands/feet"].extend([
            ("Diabetic Neuropathy", 0.95, "primary"),
            ("Type 2 Diabetes", 0.6, "complication"),
            ("Multiple Sclerosis", 0.4, "differential"),
            ("Carpal Tunnel Syndrome", 0.5, "differential")
        ])
        
        graph["Dark skin patches"].extend([
            ("Type 2 Diabetes", 0.85, "complication"),
            ("Acanthosis Nigricans", 0.9, "primary")
        ])
        
        # Additional symptoms
        graph["Increased thirst"].extend([
            ("Type 2 Diabetes", 0.8, "primary"),
            ("Dehydration", 0.7, "differential")
        ])
        
        graph["Dry mouth"].extend([
            ("Type 2 Diabetes", 0.6, "common"),
            ("Sjogren's Syndrome", 0.8, "differential")
        ])
        
        graph["Headaches"].extend([
            ("Hypertension", 0.7, "primary"),
            ("Migraine", 0.8, "differential")
        ])
        
        graph["Difficulty concentrating"].extend([
            ("Hypoglycemia", 0.7, "primary"),
            ("ADHD", 0.6, "differential")
        ])
        
        graph["Mood changes"].extend([
            ("Hypoglycemia", 0.6, "common"),
            ("Depression", 0.7, "differential")
        ])
        
        graph["Vision problems"].extend([
            ("Diabetic Retinopathy", 0.9, "primary"),
            ("Cataracts", 0.7, "differential")
        ])
        
        graph["Chest pain"].extend([
            ("Heart Disease", 0.9, "primary"),
            ("Angina", 0.85, "differential")
        ])
        
        graph["Shortness of breath"].extend([
            ("Heart Disease", 0.8, "primary"),
            ("Asthma", 0.7, "differential")
        ])
        
        graph["Swelling in legs"].extend([
            ("Heart Disease", 0.7, "complication"),
            ("Kidney Disease", 0.75, "differential")
        ])
        
        # Disease → Complication connections
        graph["Type 2 Diabetes"].extend([
            ("Diabetic Neuropathy", 0.8, "complication"),
            ("Diabetic Retinopathy", 0.75, "complication"),
            ("Diabetic Nephropathy", 0.7, "complication"),
            ("Heart Disease", 0.6, "complication"),
            ("Stroke", 0.5, "complication")
        ])
        
        graph["Type 1 Diabetes"].extend([
            ("Diabetic Ketoacidosis", 0.9, "complication"),
            ("Diabetic Neuropathy", 0.7, "complication"),
            ("Diabetic Retinopathy", 0.65, "complication")
        ])
        
        return graph
    
    def load_disease_info(self) -> Dict[str, Dict]:
        """Load detailed information about diseases"""
        return {
            "Type 2 Diabetes": {
                "description": "Chronic condition that affects the way your body metabolizes glucose.",
                "prevalence": "Common",
                "urgency": "High",
                "risk_factors": ["Obesity", "Family history", "Physical inactivity", "Age > 45"],
                "common_tests": ["Fasting blood glucose", "HbA1c", "Oral glucose tolerance test"],
                "treatments": ["Metformin", "Lifestyle changes", "Insulin therapy"],
                "complications": ["Heart disease", "Neuropathy", "Retinopathy", "Kidney disease"]
            },
            "Type 1 Diabetes": {
                "description": "Autoimmune condition where the pancreas produces little or no insulin.",
                "prevalence": "Less common",
                "urgency": "High",
                "risk_factors": ["Family history", "Genetic factors", "Environmental triggers"],
                "common_tests": ["Autoantibody tests", "C-peptide test", "Fasting blood glucose"],
                "treatments": ["Insulin therapy", "Blood sugar monitoring", "Carbohydrate counting"],
                "complications": ["Diabetic ketoacidosis", "Hypoglycemia", "Neuropathy"]
            },
            "Diabetic Neuropathy": {
                "description": "Nerve damage caused by diabetes, affecting hands and feet.",
                "prevalence": "Common in long-term diabetes",
                "urgency": "Medium",
                "risk_factors": ["Poor blood sugar control", "Long diabetes duration", "Kidney disease"],
                "common_tests": ["Neurological exam", "Nerve conduction studies", "Filament test"],
                "treatments": ["Pain management", "Blood sugar control", "Physical therapy"],
                "complications": ["Foot ulcers", "Infections", "Amputation risk"]
            },
            "Diabetic Retinopathy": {
                "description": "Eye damage caused by diabetes that can lead to blindness.",
                "prevalence": "Common in long-term diabetes",
                "urgency": "High",
                "risk_factors": ["Long diabetes duration", "Poor blood sugar control", "High blood pressure"],
                "common_tests": ["Dilated eye exam", "Fluorescein angiography", "Optical coherence tomography"],
                "treatments": ["Laser treatment", "Anti-VEGF injections", "Vitrectomy"],
                "complications": ["Vision loss", "Blindness", "Retinal detachment"]
            },
            "Heart Disease": {
                "description": "Range of conditions affecting the heart, often linked to diabetes.",
                "prevalence": "Common",
                "urgency": "High",
                "risk_factors": ["High blood pressure", "High cholesterol", "Smoking", "Diabetes"],
                "common_tests": ["ECG", "Echocardiogram", "Stress test", "Angiogram"],
                "treatments": ["Medications", "Lifestyle changes", "Surgery"],
                "complications": ["Heart attack", "Heart failure", "Stroke"]
            }
        }
    
    def load_symptom_descriptions(self) -> Dict[str, str]:
        """Load descriptions for symptoms"""
        return {
            "Frequent urination": "Needing to urinate more often than usual, especially at night",
            "Excessive thirst": "Unusual and persistent thirst that isn't relieved by drinking",
            "Extreme hunger": "Constant hunger even after eating",
            "Unexplained weight loss": "Losing weight without trying, despite eating normally or more",
            "Fatigue": "Persistent tiredness and lack of energy",
            "Blurred vision": "Vision that is not sharp, may come and go",
            "Slow healing wounds": "Cuts and sores that take longer than normal to heal",
            "Frequent infections": "Getting infections more often, especially skin, gum, or bladder infections",
            "Tingling/numbness in hands/feet": "Pins and needles sensation or loss of feeling in extremities",
            "Dark skin patches": "Dark, velvety patches in body folds and creases",
            "Increased thirst": "Drinking more fluids than usual",
            "Dry mouth": "Persistent dry feeling in the mouth",
            "Headaches": "Persistent or severe headaches",
            "Difficulty concentrating": "Trouble focusing or brain fog",
            "Mood changes": "Irritability, anxiety, or depression",
            "Vision problems": "Any issues with eyesight including blurriness or floaters",
            "Chest pain": "Pain, pressure, or discomfort in the chest area",
            "Shortness of breath": "Difficulty breathing or feeling breathless",
            "Swelling in legs": "Fluid accumulation causing swollen ankles or legs"
        }
    
    def bfs_search(self, symptoms: List[str], max_depth: int = 3) -> List[Dict]:
        """
        Breadth-First Search traversal from symptoms to diseases
        Returns diseases sorted by confidence
        """
        visited = set()
        results = []
        queue = deque()
        
        # Start BFS from each symptom
        for symptom in symptoms:
            if symptom in self.graph:
                queue.append((symptom, 0, 1.0, [symptom]))  # (node, depth, confidence, path)
        
        while queue:
            current_node, depth, confidence, path = queue.popleft()
            
            if depth > max_depth:
                continue
                
            if current_node not in visited:
                visited.add(current_node)
                
                # Check if current node is a disease (has disease info)
                if current_node in self.disease_info:
                    results.append({
                        "disease": current_node,
                        "confidence": min(1.0, confidence),
                        "path": path.copy(),
                        "depth": depth,
                        "matched_symptoms": [s for s in symptoms if s in path],
                        "algorithm": "BFS"
                    })
                
                # Explore neighbors
                for neighbor, weight, relation_type in self.graph.get(current_node, []):
                    new_confidence = confidence * weight
                    new_path = path + [neighbor]
                    
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1, new_confidence, new_path))
        
        # Sort by confidence and remove duplicates
        unique_results = {}
        for result in results:
            disease = result["disease"]
            if disease not in unique_results or result["confidence"] > unique_results[disease]["confidence"]:
                unique_results[disease] = result
        
        return sorted(unique_results.values(), key=lambda x: x["confidence"], reverse=True)
    
    def dfs_search(self, symptoms: List[str], max_depth: int = 4) -> List[Dict]:
        """
        Depth-First Search for deeper symptom-disease relationships
        """
        def dfs_recursive(node: str, depth: int, confidence: float, 
                         path: List[str], visited: Set[str], results: List[Dict]):
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            
            if node in self.disease_info:
                results.append({
                    "disease": node,
                    "confidence": min(1.0, confidence),
                    "path": path.copy(),
                    "depth": depth,
                    "matched_symptoms": [s for s in symptoms if s in path],
                    "algorithm": "DFS"
                })
            
            for neighbor, weight, relation_type in self.graph.get(node, []):
                new_path = path + [neighbor]
                new_confidence = confidence * weight
                dfs_recursive(neighbor, depth + 1, new_confidence, new_path, visited, results)
            
            visited.remove(node)  # Backtrack
        
        results = []
        visited = set()
        
        for symptom in symptoms:
            if symptom in self.graph:
                dfs_recursive(symptom, 0, 1.0, [symptom], visited, results)
        
        # Remove duplicates
        unique_results = {}
        for result in results:
            disease = result["disease"]
            if disease not in unique_results or result["confidence"] > unique_results[disease]["confidence"]:
                unique_results[disease] = result
        
        return sorted(unique_results.values(), key=lambda x: x["confidence"], reverse=True)
    
    def constrained_search(self, symptoms: List[str], constraints: Dict) -> List[Dict]:
        """
        Constraint Satisfaction Problem approach
        """
        # Get all diseases reachable from symptoms
        all_results = self.bfs_search(symptoms, max_depth=4)
        
        valid_results = []
        
        for result in all_results:
            disease = result["disease"]
            info = self.disease_info.get(disease, {})
            
            satisfies = True
            
            # Constraint 1: Minimum symptom match
            min_symptoms = constraints.get("min_symptoms", 2)
            if len(result["matched_symptoms"]) < min_symptoms:
                satisfies = False
            
            # Constraint 2: Age compatibility
            patient_age = constraints.get("age", 30)
            if "Type 2 Diabetes" in disease and patient_age < 30:
                satisfies = False  # Less common in young people
            if "Type 1 Diabetes" in disease and patient_age > 40:
                satisfies = False  # Usually diagnosed in youth
            
            # Constraint 3: Gender compatibility
            patient_gender = constraints.get("gender", "both")
            if patient_gender != "both":
                # Some diseases have gender preferences
                if "Gestational Diabetes" in disease and patient_gender != "female":
                    satisfies = False
            
            # Constraint 4: Risk factors
            risk_factors = constraints.get("risk_factors", [])
            if risk_factors:
                disease_risk_factors = info.get("risk_factors", [])
                match_score = len(set(risk_factors) & set(disease_risk_factors)) / len(disease_risk_factors)
                if match_score < 0.3:  # At least 30% match
                    satisfies = False
            
            if satisfies:
                # Calculate constrained confidence
                base_confidence = result["confidence"]
                
                # Adjust for constraints
                age_factor = 1.0
                if "Type 2 Diabetes" in disease and 45 <= patient_age <= 65:
                    age_factor = 1.2  # Higher confidence in typical age range
                
                # Adjust for symptom count
                symptom_factor = min(1.5, 1.0 + (len(result["matched_symptoms"]) / 10))
                
                constrained_confidence = min(1.0, base_confidence * age_factor * symptom_factor)
                
                result["confidence"] = constrained_confidence
                result["algorithm"] = "CSP"
                result["constraints_applied"] = constraints
                
                valid_results.append(result)
        
        return sorted(valid_results, key=lambda x: x["confidence"], reverse=True)
    
    def get_symptom_suggestions(self, query: str) -> List[str]:
        """Get symptom suggestions based on query"""
        query = query.lower()
        suggestions = []
        
        for symptom in self.graph.keys():
            if query in symptom.lower():
                suggestions.append(symptom)
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def get_disease_details(self, disease_name: str) -> Dict:
        """Get detailed information about a disease"""
        if disease_name in self.disease_info:
            info = self.disease_info[disease_name].copy()
            info["name"] = disease_name
            info["related_symptoms"] = self._get_related_symptoms(disease_name)
            return info
        return {"error": "Disease not found"}
    
    def _get_related_symptoms(self, disease: str) -> List[str]:
        """Get symptoms related to a disease"""
        symptoms = []
        for symptom, connections in self.graph.items():
            for connected_disease, _, _ in connections:
                if connected_disease == disease:
                    symptoms.append(symptom)
                    break
        return symptoms
    
    def get_all_symptoms(self) -> List[str]:
        """Get all available symptoms"""
        return list(self.graph.keys())

# Flask integration
class SymptomSearchAPI:
    def __init__(self):
        self.mapper = SymptomToDiseaseMapper()
    
    def register_routes(self, app):
        @app.route('/symptom_search')
        def symptom_search_page():
            symptoms = self.mapper.get_all_symptoms()
            return render_template('symptom_search.html', symptoms=symptoms)
        
        @app.route('/api/symptom_search', methods=['POST'])
        def search_diseases():
            try:
                data = request.json
                symptoms = data.get('symptoms', [])
                algorithm = data.get('algorithm', 'bfs')
                constraints = data.get('constraints', {})
                
                if not symptoms:
                    return jsonify({"error": "No symptoms provided"}), 400
                
                # Filter out symptoms not in graph
                valid_symptoms = [s for s in symptoms if s in self.mapper.graph]
                
                if not valid_symptoms:
                    return jsonify({"error": "No valid symptoms provided"}), 400
                
                # Perform search
                if algorithm == 'bfs':
                    results = self.mapper.bfs_search(valid_symptoms)
                elif algorithm == 'dfs':
                    results = self.mapper.dfs_search(valid_symptoms)
                elif algorithm == 'csp':
                    results = self.mapper.constrained_search(valid_symptoms, constraints)
                else:
                    results = self.mapper.bfs_search(valid_symptoms)
                
                # Add additional information
                for result in results:
                    disease_info = self.mapper.get_disease_details(result["disease"])
                    result.update({
                        "description": disease_info.get("description", ""),
                        "urgency": disease_info.get("urgency", "Medium"),
                        "recommended_tests": disease_info.get("common_tests", []),
                        "treatments": disease_info.get("treatments", []),
                        "complications": disease_info.get("complications", [])
                    })
                
                return jsonify({
                    "status": "success",
                    "results": results[:10],  # Top 10 results
                    "algorithm_used": algorithm,
                    "symptoms_searched": valid_symptoms,
                    "total_matches": len(results)
                })
                
            except Exception as e:
                app.logger.error(f"Symptom search error: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/symptom_suggestions', methods=['GET'])
        def symptom_suggestions():
            query = request.args.get('q', '')
            suggestions = self.mapper.get_symptom_suggestions(query)
            return jsonify({"suggestions": suggestions})
        
        @app.route('/api/disease_details/<disease_name>')
        def disease_details(disease_name):
            details = self.mapper.get_disease_details(disease_name)
            return jsonify(details)
        
        @app.route('/api/all_symptoms')
        def all_symptoms():
            symptoms = self.mapper.get_all_symptoms()
            return jsonify({"symptoms": symptoms})

# Helper function to integrate with Flask app
def integrate_symptom_search(app):
    symptom_api = SymptomSearchAPI()
    symptom_api.register_routes(app)