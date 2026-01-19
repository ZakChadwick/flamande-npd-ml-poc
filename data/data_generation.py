"""
Synthetic NPD Data Generator for Butchery & Food Service

Generates realistic new product development launches with:
- Product characteristics
- Market context
- Marketing factors
- Success outcomes

Based on industry patterns and realistic business logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np. random.seed(42)

def generate_npd_data(n_products=300):
    """
    Generate synthetic NPD dataset
    
    Args:
        n_products: Number of product launches to generate
        
    Returns: 
        pandas.DataFrame with NPD data
    """
    
    products = []
    
    for i in range(n_products):
        # Product characteristics
        protein_type = np. random.choice(
            ['beef', 'pork', 'lamb', 'chicken', 'mixed'], 
            p=[0.3, 0.25, 0.15, 0.2, 0.1]
        )
        
        preparation = np.random.choice(
            ['raw', 'marinated', 'seasoned', 'ready-to-cook', 'cooked'],
            p=[0.2, 0.25, 0.2, 0.25, 0.1]
        )
        
        price_point = np.random. choice(
            ['economy', 'mid-range', 'premium'],
            p=[0.3, 0.5, 0.2]
        )
        
        portion_size_g = np.random.choice([200, 250, 300, 400, 500])
        shelf_life_days = np.random.randint(3, 21)
        
        # Market context
        launch_quarter = np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])
        target_channel = np.random.choice(
            ['food_service', 'retail_butcher', 'supermarket', 'direct'],
            p=[0.4, 0.3, 0.2, 0.1]
        )
        competitive_products = np.random.randint(0, 15)
        trend_alignment = np.random.uniform(0, 1)
        
        # Marketing factors
        marketing_spend_gbp = np.random.randint(1000, 50000)
        has_sustainability_claim = np.random.choice([0, 1], p=[0.6, 0.4])
        has_origin_story = np. random.choice([0, 1], p=[0.5, 0.5])
        packaging_innovation = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Development factors
        development_time_months = np.random.randint(2, 18)
        testing_iterations = np.random.randint(1, 8)
        consultant_involved = np.random.choice([0, 1], p=[0.4, 0.6])
        
        # Calculate success probability based on realistic business logic
        success_score = 0.3  # Base probability
        
        # Premium products in food service do well
        if price_point == 'premium' and target_channel == 'food_service':
            success_score += 0.25
        
        # Marinated/ready-to-cook trend
        if preparation in ['marinated', 'ready-to-cook']:
            success_score += 0.20
        elif preparation == 'raw':
            success_score -= 0.10
        
        # High competition is bad
        success_score -= (competitive_products / 40)
        
        # Trend alignment matters
        success_score += trend_alignment * 0.15
        
        # Marketing investment helps (diminishing returns)
        success_score += min(marketing_spend_gbp / 150000, 0.12)
        
        # Sustainability is increasingly valued
        if has_sustainability_claim:
            success_score += 0.12
        
        # Origin story adds authenticity
        if has_origin_story:
            success_score += 0.08
        
        # Longer shelf life = better distribution
        success_score += (shelf_life_days / 80)
        
        # Packaging innovation
        if packaging_innovation:
            success_score += 0.08
        
        # Consultant involvement (expertise helps)
        if consultant_involved: 
            success_score += 0.10
        
        # Channel-specific adjustments
        if target_channel == 'food_service':
            success_score += 0.05
        elif target_channel == 'supermarket':
            success_score -= 0.05  # Harder to break into
        
        # Seasonal considerations
        if launch_quarter in ['Q2', 'Q3'] and protein_type in ['beef', 'pork']: 
            success_score += 0.05  # BBQ season
        
        # Add realistic noise
        success_score += np.random.normal(0, 0.12)
        
        # Clip to [0, 1]
        success_score = max(0, min(1, success_score))
        
        # Determine binary success (with some randomness)
        success = 1 if success_score > 0.5 else 0
        
        # Calculate first year revenue (correlated with success)
        if success:
            base_revenue = np.random.normal(90000, 35000)
        else:
            base_revenue = np.random. normal(25000, 15000)
        
        # Premium products have higher revenue potential
        if price_point == 'premium':
            base_revenue *= 1.4
        elif price_point == 'economy': 
            base_revenue *= 0.8
        
        first_year_revenue_gbp = max(0, int(base_revenue))
        
        # Assemble product record
        product = {
            'product_id': f'NPD_{i:03d}',
            'protein_type': protein_type,
            'preparation':  preparation,
            'price_point': price_point,
            'portion_size_g': portion_size_g,
            'shelf_life_days': shelf_life_days,
            'launch_quarter': launch_quarter,
            'target_channel': target_channel,
            'competitive_products':  competitive_products,
            'trend_alignment': round(trend_alignment, 3),
            'marketing_spend_gbp': marketing_spend_gbp,
            'has_sustainability_claim': has_sustainability_claim,
            'has_origin_story': has_origin_story,
            'packaging_innovation': packaging_innovation,
            'development_time_months': development_time_months,
            'testing_iterations': testing_iterations,
            'consultant_involved': consultant_involved,
            'success':  success,
            'first_year_revenue_gbp': first_year_revenue_gbp,
            'success_score': round(success_score, 3)  # Hidden - for analysis only
        }
        
        products.append(product)
    
    df = pd.DataFrame(products)
    
    return df


def main():
    """Generate and save synthetic NPD data"""
    
    print("Generating synthetic NPD data...")
    df = generate_npd_data(n_products=300)
    
    # Display summary statistics
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Total products:  {len(df)}")
    print(f"Success rate: {df['success'].mean():.1%}")
    print(f"Average first-year revenue (successful): £{df[df['success']==1]['first_year_revenue_gbp'].mean():,.0f}")
    print(f"Average first-year revenue (failed): £{df[df['success']==0]['first_year_revenue_gbp'].mean():,.0f}")
    
    print(f"\n{'='*60}")
    print(f"Product Distribution")
    print(f"{'='*60}")
    print(df['protein_type'].value_counts())
    print(f"\n{df['preparation'].value_counts()}")
    print(f"\n{df['price_point'].value_counts()}")
    print(f"\n{df['target_channel'].value_counts()}")
    
    # Save to CSV
    output_path = 'data/synthetic_npd_data. csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Data saved to:  {output_path}")
    
    return df


if __name__ == "__main__":
    main()