import cv2
import pytesseract
import requests
import numpy as np
import json
import os
from typing import Dict, List, Union, Tuple
import re
import time
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NutritionAnalyzer:
    def __init__(self):
        """Initialize the nutrition analyzer with necessary APIs"""
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Initialize API keys (get from environment variables)
        self.usda_api_key = os.getenv('USDA_API_KEY', '')

        # Initialize ML models
        self._initialize_models()  # Add this line

        # Load preset responses for offline use
        self.preset_responses = self._load_preset_responses()
        
        # Check if API key is available
        self.apis_available = bool(self.usda_api_key)
        if not self.apis_available:
            logger.warning("API key not found. Running in offline mode with limited functionality.")
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.nutritionix_app_id = os.getenv('NUTRITIONIX_APP_ID', '')
        self.nutritionix_api_key = os.getenv('NUTRITIONIX_API_KEY', '')

    def _check_apis(self) -> bool:
        """Check if API keys are available"""
        return bool(
            self.usda_api_key and 
            self.openai_api_key and 
            self.nutritionix_app_id and 
            self.nutritionix_api_key
        )

    def _load_preset_responses(self) -> Dict:
        """Load preset responses for offline mode"""
        # This would ideally load from a file, but for this example we'll define some responses directly
        return {
            'high_sugar': [
                "This product contains high sugar content which may contribute to blood sugar spikes.",
                "Consider alternatives with lower sugar content for better metabolic health.",
                "High sugar intake is associated with increased risk of type 2 diabetes and obesity."
            ],
            'high_sodium': [
                "The sodium content in this product is high, which may impact blood pressure.",
                "Consider lower sodium alternatives, especially if you have hypertension.",
                "High sodium intake is linked to increased risk of cardiovascular disease."
            ],
            'high_fat': [
                "This product is high in total fat, which contributes to higher calorie content.",
                "Check the types of fat - saturated and trans fats should be limited for heart health.",
                "Consider products with healthier fat profiles, like those containing unsaturated fats."
            ],
            'low_protein': [
                "This product is relatively low in protein, which is important for muscle maintenance.",
                "Consider pairing with protein-rich foods for a more balanced meal.",
                "Adequate protein intake supports tissue repair and immune function."
            ],
            'general': [
                "Read food labels carefully to make informed dietary choices.",
                "Compare similar products to find options with better nutritional profiles.",
                "Consider portion sizes when evaluating nutritional content."
            ]
        }

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Tesseract OCR"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Preprocess image to improve OCR accuracy
            preprocessed = self._preprocess_image(image)
            
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(preprocessed)
            
            logger.info(f"Successfully extracted text from image")
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening

    def analyze_nutrition(self, text: str) -> Dict:
        """Extract nutrition facts from OCR text"""
        # Parse text to extract nutrition data using regex
        patterns = {
            'Calories': r'Calories\s*(\d+)',
            'Total Fat': r'Total Fat\s*(\d+(?:\.\d+)?)g',
            'Saturated Fat': r'Saturated Fat\s*(\d+(?:\.\d+)?)g',
            'Trans Fat': r'Trans Fat\s*(\d+(?:\.\d+)?)g',
            'Cholesterol': r'Cholesterol\s*(\d+)mg',
            'Sodium': r'Sodium\s*(\d+)mg',
            'Total Carbohydrate': r'Total Carbohydrate[s]?\s*(\d+)g',
            'Dietary Fiber': r'Dietary Fiber\s*(\d+)g',
            'Sugars': r'Sugars\s*(\d+)g',
            'Added Sugars': r'Added Sugars\s*(\d+)g',
            'Protein': r'Protein\s*(\d+)g',
            'Vitamin D': r'Vitamin D\s*(\d+)',
            'Calcium': r'Calcium\s*(\d+)',
            'Iron': r'Iron\s*(\d+)',
            'Potassium': r'Potassium\s*(\d+)'
        }
        
        nutrition_data = {}
        
        # Extract values using regex
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    nutrition_data[nutrient] = value
                except (ValueError, IndexError):
                    pass
        
        # If using online API and we have a key, attempt to get more accurate data
        if self.apis_available and self.nutritionix_api_key:  # Fixed variable name
            try:
                # Try to get enhanced data from nutrition API
                enhanced_data = self._get_nutrition_data(text)
                if enhanced_data:
                    nutrition_data.update(enhanced_data)
            except Exception as e:
                logger.warning(f"Error fetching nutrition API data: {str(e)}. Using local parsing only.")
        
        return nutrition_data

    def _get_nutrition_data(self, text: str) -> Dict:
        try:
            food_items = self._extract_food_items(text)
            if not food_items:
                return {}

            url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                'api_key': self.usda_api_key,
                'query': food_items,
                'pageSize': 1
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                processed_data = {}
                if data.get('foods'):
                    food = data['foods'][0]
                    nutrients = food.get('foodNutrients', [])
                    
                    # Expanded nutrient mapping
                    nutrient_map = {
                        1008: 'Calories',
                        1004: 'Total Fat',
                        1258: 'Saturated Fat',
                        1257: 'Trans Fat',
                        1253: 'Cholesterol',
                        1093: 'Sodium',
                        1005: 'Total Carbohydrate',
                        1079: 'Dietary Fiber',
                        2000: 'Sugars',
                        1235: 'Added Sugars',
                        1003: 'Protein',
                        1114: 'Vitamin D',
                        1087: 'Calcium',
                        1089: 'Iron',
                        1092: 'Potassium',
                        1095: 'Zinc',
                        1106: 'Vitamin A',
                        1162: 'Vitamin C',
                        1178: 'Vitamin E',
                        1175: 'Vitamin B-6',
                        1165: 'Vitamin B-12',
                        1109: 'Vitamin K'
                    }
                    
                    for nutrient in nutrients:
                        nutrient_id = nutrient.get('nutrientId')
                        if nutrient_id in nutrient_map:
                            processed_data[nutrient_map[nutrient_id]] = nutrient.get('value', 0)
                    
                return processed_data
            else:
                logger.warning(f"USDA API returned status code {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error in USDA API: {str(e)}")
            return {}

    def _extract_food_items(self, text: str) -> str:
        """Extract food items from text for API query"""
        # Simplified - in a real app, you'd use NLP to identify food items
        # For this example, we just grab text that might be product names
        lines = text.split('\n')
        for line in lines[:5]:  # Check first few lines
            if len(line) > 3 and not any(x in line.lower() for x in ['nutrition', 'facts', 'serving']):
                return line
        return ""

    def _initialize_models(self):
        """Initialize ML models for nutrition analysis"""
        # Define feature columns for the model
        self.feature_columns = [
            'Calories', 'Total Fat', 'Saturated Fat', 'Trans Fat', 
            'Cholesterol', 'Sodium', 'Total Carbohydrate', 'Dietary Fiber',
            'Sugars', 'Protein', 'Vitamin D', 'Calcium', 'Iron', 'Potassium'
        ]
        
        # Initialize models for different health focuses
        self.models = {
            'heart_health': self._create_health_model('heart_health'),
            'weight_management': self._create_health_model('weight_management'),
            'diabetes': self._create_health_model('diabetes'),
            'fitness': self._create_health_model('fitness'),
            'general': self._create_health_model('general')
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()

    def _create_health_model(self, health_focus: str) -> RandomForestClassifier:
        """Create and train model for specific health focus"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train the model with predefined rules
        X_train, y_train = self._generate_training_data(health_focus)
        model.fit(X_train, y_train)
        
        return model

    def _generate_training_data(self, health_focus: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data based on nutritional guidelines"""
        # Generate synthetic training data based on health guidelines
        X = []
        y = []
        
        # Generate positive and negative examples
        for _ in range(1000):
            sample = np.random.rand(len(self.feature_columns))
            
            if health_focus == 'heart_health':
                label = 1 if (sample[2] < 0.3 and sample[4] < 0.3) else 0  # Low sat fat and cholesterol
            elif health_focus == 'diabetes':
                label = 1 if (sample[8] < 0.2) else 0  # Low sugar
            elif health_focus == 'weight_management':
                label = 1 if (sample[0] < 0.3 and sample[1] < 0.3) else 0  # Low calories and fat
            elif health_focus == 'fitness':
                label = 1 if (sample[9] > 0.6) else 0  # High protein
            else:  # general
                label = 1 if np.mean(sample) < 0.5 else 0
            
            X.append(sample)
            y.append(label)
        
        return np.array(X), np.array(y)

    def get_health_recommendations(self, nutrition_data: Dict, detail_level: str = 'basic', health_focus: str = 'general') -> Dict:
        """Generate recommendations using ML model"""
        # Map health focus to model's expected values
        health_focus_map = {
            'weight loss': 'weight_management',
            'weight management': 'weight_management',
            'heart health': 'heart_health',
            'general': 'general',
            'fitness': 'fitness',
            'diabetes': 'diabetes'
        }
        health_focus = health_focus_map.get(health_focus.lower(), 'general')
        
        # Prepare input data
        input_data = self._prepare_input_data(nutrition_data)
        
        # Get model predictions and probabilities
        model = self.models.get(health_focus, self.models['general'])
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Generate recommendations based on predictions
        recommendations = self._generate_ml_recommendations(
            nutrition_data,
            prediction,
            probabilities,
            health_focus,
            detail_level
        )
        
        return recommendations

    def _prepare_input_data(self, nutrition_data: Dict) -> np.ndarray:
        """Prepare input data for ML model"""
        input_vector = []
        for feature in self.feature_columns:
            input_vector.append(nutrition_data.get(feature, 0))
        return np.array([input_vector])

    def _generate_ml_recommendations(self, nutrition_data: Dict, prediction: int, 
                                  probabilities: np.ndarray, health_focus: str, 
                                  detail_level: str) -> Dict:
        """Generate recommendations based on ML predictions"""
        recommendations = {
            'General Assessment': [],
            'Specific Concerns': [],
            'Recommendations': [],
            'Health Benefits': []
        }
        
        # Generate recommendations based on prediction confidence
        confidence = max(probabilities)
        
        if prediction == 1:
            recommendations['General Assessment'].append(
                f"This food is generally suitable for {health_focus} focus (Confidence: {confidence:.2%})"
            )
        else:
            recommendations['General Assessment'].append(
                f"This food may need adjustments for {health_focus} focus (Confidence: {confidence:.2%})"
            )
        
        # Adjust detail level based on user selection
        if detail_level.lower() == 'basic':
            # Basic analysis - just show top concern and basic recommendation
            feature_importance = dict(zip(self.feature_columns, 
                                        self.models[health_focus].feature_importances_))
            top_feature, importance = max(feature_importance.items(), key=lambda x: x[1])
            value = nutrition_data.get(top_feature, 0)
            recommendations['Specific Concerns'].append(
                f"Key Factor: {top_feature}: {value}"
            )

        elif detail_level.lower() == 'detailed':
            # Detailed analysis - show top 3 concerns with importance
            feature_importance = dict(zip(self.feature_columns, 
                                        self.models[health_focus].feature_importances_))
            for feature, importance in sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True)[:3]:
                value = nutrition_data.get(feature, 0)
                recommendations['Specific Concerns'].append(
                    f"{feature}: {value} (Importance: {importance:.2%})"
                )

        else:  # comprehensive
            # Comprehensive analysis - show all relevant factors and detailed explanations
            feature_importance = dict(zip(self.feature_columns, 
                                        self.models[health_focus].feature_importances_))
            for feature, importance in sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True):
                if importance > 0.05:  # Only show factors with significant importance
                    value = nutrition_data.get(feature, 0)
                    threshold = self._get_threshold(feature, health_focus)
                    status = "High" if value > threshold else "Low"
                    recommendations['Specific Concerns'].append(
                        f"{feature}: {value} ({status}, Importance: {importance:.2%})"
                    )

        # Add health benefits and recommendations based on detail level
        self._add_health_benefits(recommendations, nutrition_data, detail_level)
        self._add_specific_recommendations(recommendations, nutrition_data, detail_level)
        
        return recommendations

    def _get_threshold(self, feature: str, health_focus: str) -> float:
        """Get threshold values for nutritional components based on health focus"""
        thresholds = {
            'Calories': {'weight_management': 300, 'general': 400},
            'Total Fat': {'heart_health': 10, 'weight_management': 8, 'general': 15},
            'Saturated Fat': {'heart_health': 3, 'general': 5},
            'Cholesterol': {'heart_health': 50, 'general': 100},
            'Sodium': {'heart_health': 400, 'general': 500},
            'Sugars': {'diabetes': 5, 'weight_management': 8, 'general': 10},
            'Protein': {'fitness': 15, 'general': 10}
        }
        
        if feature in thresholds:
            return thresholds[feature].get(health_focus, thresholds[feature]['general'])
        return 0

    def _add_health_benefits(self, recommendations: Dict, nutrition_data: Dict, detail_level: str = 'basic') -> None:
        """Add health benefits based on nutritional content and detail level"""
        benefits = []
        
        if detail_level.lower() == 'basic':
            # Basic - just show main benefit
            if nutrition_data.get('Protein', 0) > 5:
                benefits.append("Good source of protein")
        
        elif detail_level.lower() == 'detailed':
            # Detailed - show benefits with brief explanations
            if nutrition_data.get('Protein', 0) > 5:
                benefits.append("Good source of protein for muscle maintenance and repair")
            if nutrition_data.get('Dietary Fiber', 0) > 3:
                benefits.append("Contains dietary fiber which supports digestive health")
            
        else:  # comprehensive
            # Comprehensive - show all benefits with detailed explanations
            if nutrition_data.get('Protein', 0) > 5:
                benefits.append("Good source of protein for muscle maintenance, repair, and immune function")
            if nutrition_data.get('Dietary Fiber', 0) > 3:
                benefits.append("Contains dietary fiber which supports digestive health, helps maintain steady blood sugar, and promotes satiety")
            if nutrition_data.get('Calcium', 0) > 100:
                benefits.append("Good source of calcium for bone health, muscle function, and nerve signaling")
            if nutrition_data.get('Iron', 0) > 2:
                benefits.append("Contains iron which supports oxygen transport, energy production, and cognitive function")
            
        if benefits:
            recommendations['Health Benefits'].extend(benefits)
        else:
            recommendations['Health Benefits'].append(
                "This food can be part of a balanced diet. Consider combining it with other nutrient-rich foods."
            )

    def _add_specific_recommendations(self, recommendations: Dict, nutrition_data: Dict, detail_level: str = 'basic') -> None:
        """Add specific recommendations based on nutritional content and detail level"""
        if detail_level.lower() == 'basic':
            # Basic - just one key recommendation
            if nutrition_data.get('Calories', 0) > 300:
                recommendations['Recommendations'].append("Consider portion size to manage calorie intake")
            else:
                recommendations['Recommendations'].extend(self.preset_responses['general'][:1])
                
        elif detail_level.lower() == 'detailed':
            # Detailed - key recommendations with brief explanations
            if nutrition_data.get('Sodium', 0) > 400:
                recommendations['Recommendations'].extend(self.preset_responses['high_sodium'][:2])
            if nutrition_data.get('Sugars', 0) > 10:
                recommendations['Recommendations'].extend(self.preset_responses['high_sugar'][:2])
            
        else:  # comprehensive
            # Comprehensive - all relevant recommendations with detailed explanations
            if nutrition_data.get('Sodium', 0) > 400:
                recommendations['Recommendations'].extend(self.preset_responses['high_sodium'])
            if nutrition_data.get('Sugars', 0) > 10:
                recommendations['Recommendations'].extend(self.preset_responses['high_sugar'])
            if nutrition_data.get('Total Fat', 0) > 15:
                recommendations['Recommendations'].extend(self.preset_responses['high_fat'])
            if nutrition_data.get('Protein', 0) < 5:
                recommendations['Recommendations'].extend(self.preset_responses['low_protein'])
            
        if not recommendations['Recommendations']:
            recommendations['Recommendations'].extend(self.preset_responses['general'])


# For testing purposes
if __name__ == "__main__":
    analyzer = NutritionAnalyzer()
    
    # Debug API keys
    print("\n=== API Configuration Status ===")
    print(f"USDA API Key: {'✓' if analyzer.usda_api_key else '✗'}")
    print(f"OpenAI API Key: {'✓' if analyzer.openai_api_key else '✗'}")
    print(f"Nutritionix App ID: {'✓' if analyzer.nutritionix_app_id else '✗'}")
    print(f"Nutritionix API Key: {'✓' if analyzer.nutritionix_api_key else '✗'}")
    print(f"APIs available: {'✓' if analyzer.apis_available else '✗'}")
    print("\n=== Starting Analysis ===")

    # Use an absolute path to your test image
    test_image = "d:\\projects\\AI project\\test_images\\nutrition_label.jpg"
    
    if os.path.exists(test_image):
        print(f"\nProcessing image: {test_image}")
        
        # Test text extraction
        print("\n1. Testing OCR...")
        text = analyzer.extract_text(test_image)
        print("Extracted Text (first 200 chars):")
        print(text[:200] + "..." if len(text) > 200 else text)
        
        # Test nutrition analysis
        print("\n2. Testing Nutrition Analysis...")
        nutrition = analyzer.analyze_nutrition(text)
        print("Nutrition Data:")
        for k, v in nutrition.items():
            print(f"  {k}: {v}")
            
        # Test recommendations
        print("\n3. Testing Health Recommendations...")
        recommendations = analyzer.get_health_recommendations(
            nutrition,
            detail_level='detailed',
            health_focus='general'
        )
        print("Recommendations:")
        print(json.dumps(recommendations, indent=2))
    else:
        print(f"\nError: Test image not found at {test_image}")
        print("Please place a nutrition label image in the test_images folder.")