"""
Feature Engineering Utilities

Additional feature transformations and engineering for NPD prediction.
"""

import pandas as pd
import numpy as np


def create_interaction_features(df):
    """
    Create interaction features between key variables
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame: With additional interaction features
    """
    df_enhanced = df.copy()
    
    # Price x Channel interactions
    df_enhanced['premium_food_service'] = (
        (df['price_point'] == 'premium') &
        (df['target_channel'] == 'food_service')
    ).astype(int)
    
    df_enhanced['economy_supermarket'] = (
        (df['price_point'] == 'economy') &
        (df['target_channel'] == 'supermarket')
    ).astype(int)
    
    # Preparation x Protein interactions
    df_enhanced['marinated_lamb'] = (
        (df['preparation'] == 'marinated') &
        (df['protein_type'] == 'lamb')
    ).astype(int)
    
    df_enhanced['ready_to_cook_chicken'] = (
        (df['preparation'] == 'ready-to-cook') &
        (df['protein_type'] == 'chicken')
    ).astype(int)
    
    # Marketing effectiveness (spend relative to price point)
    price_point_map = {'economy': 1, 'mid-range': 2, 'premium': 3}
    df_enhanced['marketing_per_price_tier'] = (
        df['marketing_spend_gbp'] / df['price_point'].map(price_point_map)
    )
    
    # Competitive intensity
    df_enhanced['competition_level'] = pd.cut(
        df['competitive_products'],
        bins=[0, 3, 7, 15],
        labels=['low', 'medium', 'high']
    )
    
    # Product readiness score
    df_enhanced['readiness_score'] = (
        df['testing_iterations'] / df['development_time_months']
    )
    
    # Premium positioning flag
    df_enhanced['is_premium_positioned'] = (
        (df['price_point'] == 'premium') &
        ((df['has_sustainability_claim'] == 1) | (df['has_origin_story'] == 1))
    ).astype(int)
    
    # Convenience score
    convenience_map = {
        'raw': 0,
        'seasoned': 1,
        'marinated': 2,
        'ready-to-cook': 3,
        'cooked': 4
    }
    df_enhanced['convenience_score'] = df['preparation'].map(convenience_map)
    
    return df_enhanced


def create_aggregated_features(df):
    """
    Create aggregated features by groups
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame: With aggregated features
    """
    df_enhanced = df.copy()
    
    # Average success rate by protein type
    protein_success = df.groupby('protein_type')['success'].transform('mean')
    df_enhanced['protein_avg_success'] = protein_success
    
    # Average success rate by channel
    channel_success = df.groupby('target_channel')['success'].transform('mean')
    df_enhanced['channel_avg_success'] = channel_success
    
    # Average success rate by preparation
    prep_success = df.groupby('preparation')['success'].transform('mean')
    df_enhanced['preparation_avg_success'] = prep_success
    
    return df_enhanced


def create_temporal_features(df, launch_date_col='launch_quarter'):
    """
    Create temporal features
    
    Args:
        df: DataFrame with temporal information
        launch_date_col: Column name for launch timing

    Returns:
        DataFrame: With temporal features
    """
    df_enhanced = df.copy()

    # BBQ season flag (Q2, Q3)
    df_enhanced['is_bbq_season'] = df[launch_date_col].isin(['Q2', 'Q3']).astype(int)
    
    # Holiday season flag (Q4)
    df_enhanced['is_holiday_season'] = (df[launch_date_col] == 'Q4').astype(int)
    
    # Post-holiday flag (Q1)
    df_enhanced['is_post_holiday'] = (df[launch_date_col] == 'Q1').astype(int)
    
    return df_enhanced


def create_risk_indicators(df):
    """
    Create risk indicator features
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame: With risk indicators
    """
    df_enhanced = df.copy()
    
    # High competition risk
    df_enhanced['high_competition_risk'] = (df['competitive_products'] > 10).astype(int)
    
    # Short shelf life risk
    df_enhanced['shelf_life_risk'] = (df['shelf_life_days'] < 7).astype(int)
    
    # Underfunded marketing risk
    df_enhanced['marketing_risk'] = (df['marketing_spend_gbp'] < 10000).astype(int)
    
    # Long development risk (may indicate challenges)
    df_enhanced['development_risk'] = (df['development_time_months'] > 12).astype(int)
    
    # No differentiation risk
    df_enhanced['differentiation_risk'] = (
        (df['has_sustainability_claim'] == 0) &
        (df['has_origin_story'] == 0) &
        (df['packaging_innovation'] == 0)
    ).astype(int)
    
    # Overall risk score (sum of risk flags)
    risk_columns = [col for col in df_enhanced.columns if '_risk' in col]
    df_enhanced['total_risk_score'] = df_enhanced[risk_columns].sum(axis=1)
    
    return df_enhanced


def normalize_numeric_features(df, numeric_cols=None):
    """
    Normalize numeric features to 0-1 scale
    
    Args:
        df: DataFrame with numeric features
        numeric_cols: List of columns to normalize (None = auto-detect)
        
    Returns:
        DataFrame: With normalized features
    """
    df_normalized = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val > min_val:
            df_normalized[f'{col}_normalized'] = (
                (df[col] - min_val) / (max_val - min_val)
            )
    
    return df_normalized


def create_all_features(df):
    """
    Apply all feature engineering transformations
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame: With all engineered features
    """
    df_enhanced = df.copy()
    
    df_enhanced = create_interaction_features(df_enhanced)
    df_enhanced = create_temporal_features(df_enhanced)
    df_enhanced = create_risk_indicators(df_enhanced)
    
    return df_enhanced


def main():
    """Demo feature engineering"""
    
    print("="*60)
    print("Feature Engineering Demo")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/synthetic_npd_data.csv')
    print(f"\nOriginal features:  {len(df.columns)}")
    
    # Apply feature engineering
    df_enhanced = create_all_features(df)
    print(f"Enhanced features: {len(df_enhanced.columns)}")
    
    # Show new features
    new_features = set(df_enhanced.columns) - set(df.columns)
    print(f"\n{len(new_features)} new features created:")
    for feature in sorted(new_features):
        print(f"  - {feature}")
    
    # Show sample
    print("\nSample of enhanced data:")
    print(df_enhanced[['product_id', 'premium_food_service', 'convenience_score', 
                       'total_risk_score', 'is_bbq_season']].head())


if __name__ == "__main__":
    main()