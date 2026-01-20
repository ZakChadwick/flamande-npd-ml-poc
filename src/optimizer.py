"""
NPD Product Optimization Engine

Recommends product modifications to improve success probability
using trained ML model.
"""

import pandas as pd
import numpy as np
from itertools import product as itertools_product


class NPDOptimizer:
    """
    Product concept optimizer
    
    Analyzes product concepts and recommends modifications to
    improve market success probability.
    """
    
    def __init__(self, predictor):
        """
        Initialize optimizer with trained predictor
        
        Args:
            predictor: Trained NPDPredictor instance
        """
        self.predictor = predictor
        
    def predict_success(self, concept):
        """
        Predict success probability for a concept
        
        Args:
            concept: dict with product attributes
            
        Returns:
            float: Success probability
        """
        return self.predictor.predict_single(concept)
    
    def optimize_product_concept(self, base_concept, constraints=None, top_n=5):
        """
        Find best modifications to improve success probability
        
        Args:
            base_concept: dict with current product attributes
            constraints: dict with optimization constraints
            top_n: Number of top recommendations to return
            
        Returns:
            list: Top recommendations with impact scores
        """
        if constraints is None:
            constraints = {}
        
        # Get baseline probability
        baseline_prob = self.predict_success(base_concept)

        recommendations = []
        
        # Test preparation variations
        if not constraints.get('fixed_preparation'):
            for prep in ['raw', 'marinated', 'seasoned', 'ready-to-cook', 'cooked']:
                if prep != base_concept.get('preparation'):
                    test_concept = base_concept.copy()
                    test_concept['preparation'] = prep
                    prob = self.predict_success(test_concept)
                    
                    if prob > baseline_prob:
                        recommendations.append({
                            'change': f"Change preparation to '{prep}'",
                            'category': 'preparation',
                            'new_value': prep,
                            'baseline_probability': baseline_prob,
                            'new_probability': prob,
                            'lift': prob - baseline_prob,
                            'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100
                        })
        
        # Test protein type variations
        if not constraints.get('fixed_protein'):
            for protein in ['beef', 'pork', 'lamb', 'chicken', 'mixed']:
                if protein != base_concept.get('protein_type'):
                    test_concept = base_concept.copy()
                    test_concept['protein_type'] = protein
                    prob = self.predict_success(test_concept)
                    
                    if prob > baseline_prob:
                        recommendations.append({
                            'change': f"Switch protein to '{protein}'",
                            'category': 'protein_type',
                            'new_value': protein,
                            'baseline_probability': baseline_prob,
                            'new_probability': prob,
                            'lift': prob - baseline_prob,
                            'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100
                        })
        
        # Test price point variations
        if not constraints.get('fixed_price'):
            for price in ['economy', 'mid-range', 'premium']:
                if price != base_concept.get('price_point'):
                    test_concept = base_concept.copy()
                    test_concept['price_point'] = price
                    prob = self.predict_success(test_concept)

                    if prob > baseline_prob:
                        recommendations.append({
                            'change': f"Reposition as '{price}' price point",
                            'category': 'price_point',
                            'new_value': price,
                            'baseline_probability': baseline_prob,
                            'new_probability': prob,
                            'lift': prob - baseline_prob,
                            'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100
                        })
        
        # Test target channel variations
        if not constraints.get('fixed_channel'):
            for channel in ['food_service', 'retail_butcher', 'supermarket', 'direct']:
                if channel != base_concept.get('target_channel'):
                    test_concept = base_concept.copy()
                    test_concept['target_channel'] = channel
                    prob = self.predict_success(test_concept)

                    if prob > baseline_prob:
                        recommendations.append({
                            'change': f"Target '{channel.replace('_', ' ')}' channel",
                            'category': 'target_channel',
                            'new_value': channel,
                            'baseline_probability': baseline_prob,
                            'new_probability': prob,
                            'lift': prob - baseline_prob,
                            'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100
                        })
        
        # Test adding sustainability claim
        if not base_concept.get('has_sustainability_claim'):
            test_concept = base_concept.copy()
            test_concept['has_sustainability_claim'] = 1
            prob = self.predict_success(test_concept)
            
            if prob > baseline_prob:
                recommendations.append({
                    'change': "Add sustainability claim",
                    'category': 'sustainability',
                    'new_value': 1,
                    'baseline_probability': baseline_prob,
                    'new_probability': prob,
                    'lift': prob - baseline_prob,
                    'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100
                })
        
        # Test adding origin story
        if not base_concept.get('has_origin_story'):
            test_concept = base_concept.copy()
            test_concept['has_origin_story'] = 1
            prob = self.predict_success(test_concept)
            
            if prob > baseline_prob:
                recommendations.append({
                    'change': "Add origin/authenticity story",
                    'category': 'origin_story',
                    'new_value': 1,
                    'baseline_probability': baseline_prob,
                    'new_probability': prob,
                    'lift': prob - baseline_prob,
                    'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100
                })
        
        # Test adding packaging innovation
        if not base_concept.get('packaging_innovation'):
            test_concept = base_concept.copy()
            test_concept['packaging_innovation'] = 1
            prob = self.predict_success(test_concept)
            
            if prob > baseline_prob:
                recommendations.append({
                    'change': "Add innovative packaging features",
                    'category': 'packaging',
                    'new_value': 1,
                    'baseline_probability': baseline_prob,
                    'new_probability': prob,
                    'lift': prob - baseline_prob,
                    'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100
                })
        
        # Test increasing marketing spend (if within constraints)
        max_marketing_increase = constraints.get('max_marketing_increase_pct', 50)
        current_marketing = base_concept.get('marketing_spend_gbp', 0)
        
        if current_marketing > 0:
            for increase_pct in [25, 50]:
                if increase_pct <= max_marketing_increase:
                    test_concept = base_concept.copy()
                    new_marketing = int(current_marketing * (1 + increase_pct/100))
                    test_concept['marketing_spend_gbp'] = new_marketing
                    prob = self.predict_success(test_concept)
                    
                    if prob > baseline_prob:
                        recommendations.append({
                            'change': f"Increase marketing budget by {increase_pct}%",
                            'category': 'marketing',
                            'new_value': new_marketing,
                            'baseline_probability': baseline_prob,
                            'new_probability': prob,
                            'lift': prob - baseline_prob,
                            'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100,
                            'investment_required': new_marketing - current_marketing
                        })
        
        # Sort by lift and return top N
        recommendations.sort(key=lambda x: x['lift'], reverse=True)
        
        return recommendations[:top_n]

    def optimize_multi_factor(self, base_concept, factors_to_optimize, constraints=None):
        """
        Optimize multiple factors simultaneously
        
        Args:
            base_concept: dict with current product attributes
            factors_to_optimize: list of factors to test combinations of
            constraints: dict with optimization constraints
            
        Returns:
            dict: Best combination found
        """
        if constraints is None:
            constraints = {}
        
        baseline_prob = self.predict_success(base_concept)
        
        # Define options for each factor
        factor_options = {
            'preparation': ['raw', 'marinated', 'seasoned', 'ready-to-cook', 'cooked'],
            'price_point': ['economy', 'mid-range', 'premium'],
            'target_channel': ['food_service', 'retail_butcher', 'supermarket', 'direct'],
            'has_sustainability_claim': [0, 1],
            'has_origin_story': [0, 1],
            'packaging_innovation': [0, 1]
        }
        
        # Get options for specified factors
        options_to_test = {}
        for factor in factors_to_optimize:
            if factor in factor_options:
                options_to_test[factor] = factor_options[factor]
        
        # Generate all combinations
        factor_names = list(options_to_test.keys())
        factor_values = list(options_to_test.values())
        
        best_concept = None
        best_prob = baseline_prob
        
        # Test combinations (limit to avoid combinatorial explosion)
        for combination in itertools_product(*factor_values):
            test_concept = base_concept.copy()
            
            # Apply combination
            for i, factor_name in enumerate(factor_names):
                test_concept[factor_name] = combination[i]
            
            prob = self.predict_success(test_concept)
            
            if prob > best_prob:
                best_prob = prob
                best_concept = test_concept.copy()

        if best_concept:
            return {
                'optimized_concept': best_concept,
                'baseline_probability': baseline_prob,
                'optimized_probability': best_prob,
                'lift': best_prob - baseline_prob,
                'relative_lift_pct': ((best_prob - baseline_prob) / baseline_prob) * 100,
                'changes_made': {k: best_concept[k] for k in factor_names}
            }
        else:
            return {
                'message': 'No improvement found',
                'baseline_probability': baseline_prob
            }
    
    def scenario_analysis(self, base_concept, scenarios):
        """
        Compare multiple predefined scenarios
        
        Args:
            base_concept: dict with base product attributes
            scenarios: list of dicts with scenario modifications
            
        Returns:
            DataFrame: Scenario comparison
        """
        results = []
        
        # Baseline
        baseline_prob = self.predict_success(base_concept)
        results.append({
            'scenario': 'Baseline',
            'success_probability': baseline_prob,
            'lift': 0,
            'relative_lift_pct': 0
        })
        
        # Test each scenario
        for i, scenario in enumerate(scenarios, 1):
            test_concept = base_concept.copy()
            test_concept.update(scenario)
            
            prob = self.predict_success(test_concept)
            
            results.append({
                'scenario': f"Scenario {i}",
                'success_probability': prob,
                'lift': prob - baseline_prob,
                'relative_lift_pct': ((prob - baseline_prob) / baseline_prob) * 100 if baseline_prob > 0 else 0,
                'modifications': scenario
            })
        
        return pd.DataFrame(results)


def main():
    """Demo optimization capabilities"""
    from model import NPDPredictor
    
    print("="*60)
    print("NPD Product Optimization Demo")
    print("="*60)
    
    # Load trained model
    print("\n📂 Loading trained model...")
    predictor = NPDPredictor.load('models/npd_predictor.pkl')
    print("✅ Model loaded")
    
    # Initialize optimizer
    optimizer = NPDOptimizer(predictor)
    
    # Define test concept
    test_concept = {
        'protein_type': 'beef',
        'preparation': 'raw',
        'price_point': 'mid-range',
        'portion_size_g': 250,
        'shelf_life_days': 7,
        'launch_quarter': 'Q2',
        'target_channel': 'retail_butcher',
        'competitive_products': 8,
        'trend_alignment': 0.6,
        'marketing_spend_gbp': 15000,
        'has_sustainability_claim': 0,
        'has_origin_story': 1,
        'packaging_innovation': 0,
        'development_time_months': 6,
        'testing_iterations': 3,
        'consultant_involved': 1
    }
    
    print("\n" + "="*60)
    print("BASE CONCEPT")
    print("="*60)
    print(f"Product: {test_concept['preparation'].title()} {test_concept['protein_type'].title()}")
    print(f"Price: {test_concept['price_point'].title()}")
    print(f"Channel: {test_concept['target_channel'].replace('_', ' ').title()}")
    print(f"Marketing: £{test_concept['marketing_spend_gbp']:,}")
    
    baseline_prob = optimizer.predict_success(test_concept)
    print(f"\n🎯 Baseline Success Probability: {baseline_prob:.1%}")
    
    # Get recommendations
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    recommendations = optimizer.optimize_product_concept(test_concept, top_n=5)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['change']}")
            print(f"   New Success Probability: {rec['new_probability']:.1%}")
            print(f"   Improvement: +{rec['lift']*100:.1f} percentage points ({rec['relative_lift_pct']:.1f}% relative)")
            
            if 'investment_required' in rec:
                print(f"   Investment Required: £{rec['investment_required']:,}")
    else:
        print("✅ This concept is already highly optimized!")
    
    # Multi-factor optimization
    print("\n" + "="*60)
    print("MULTI-FACTOR OPTIMIZATION")
    print("="*60)
    
    multi_result = optimizer.optimize_multi_factor(
        test_concept,
        factors_to_optimize=['preparation', 'has_sustainability_claim', 'packaging_innovation']
    )
    
    if 'optimized_concept' in multi_result:
        print(f"\nBest combination found:")
        print(f"Success Probability: {multi_result['optimized_probability']:.1%}")
        print(f"Improvement: +{multi_result['lift']*100:.1f} percentage points")
        print(f"\nChanges:")
        for factor, value in multi_result['changes_made'].items():
            print(f"  - {factor}: {value}")

    # Scenario analysis
    print("\n" + "="*60)
    print("SCENARIO ANALYSIS")
    print("="*60)
    
    scenarios = [
        {
            'preparation': 'marinated',
            'has_sustainability_claim': 1
        },
        {
            'price_point': 'premium',
            'target_channel': 'food_service',
            'has_origin_story': 1
        },
        {
            'preparation': 'ready-to-cook',
            'packaging_innovation': 1,
            'marketing_spend_gbp': 25000
        }
    ]
    
    scenario_results = optimizer.scenario_analysis(test_concept, scenarios)
    print("\n", scenario_results[['scenario', 'success_probability', 'relative_lift_pct']])


if __name__ == "__main__":
    main()