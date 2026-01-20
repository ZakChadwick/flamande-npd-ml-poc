"""
LLM-Powered NPD Data Generation

Uses generative AI (OpenAI GPT-4 or Anthropic Claude) to create realistic 
synthetic training data for NPD success prediction.

This supplements the rule-based data generation with AI-generated data that
leverages LLM knowledge of real-world food industry patterns.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Import prompt templates
from data.llm_prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    get_single_product_prompt,
    get_batch_product_prompt,
    get_validation_prompt,
    VALIDATION_RULES
)

# Load environment variables
load_dotenv()


class LLMDataGenerator:
    """
    LLM-powered data generator for NPD product launches.
    
    Supports OpenAI (GPT-4) and Anthropic (Claude) APIs.
    """
    
    def __init__(
        self, 
        provider: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.8
    ):
        """
        Initialize LLM data generator.
        
        Args:
            provider: 'openai' or 'anthropic' (defaults to env LLM_PROVIDER)
            api_key: API key (defaults to env OPENAI_API_KEY or ANTHROPIC_API_KEY)
            model: Model name (defaults to gpt-4 or claude-3-sonnet-20240229)
            temperature: Sampling temperature for variety (default 0.8)
        """
        self.provider = provider or os.getenv('LLM_PROVIDER', 'openai')
        self.temperature = temperature
        
        # Initialize API client
        if self.provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
                self.model = model or 'gpt-4'
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai>=1.0.0")
            except Exception as e:
                raise ValueError(f"Failed to initialize OpenAI client: {e}")
                
        elif self.provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
                self.model = model or 'claude-3-sonnet-20240229'
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic>=0.18.0")
            except Exception as e:
                raise ValueError(f"Failed to initialize Anthropic client: {e}")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Call LLM API with retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt defining role
            
        Returns:
            LLM response text
        """
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
            
        elif self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
    
    def generate_single_product_with_llm(
        self, 
        context: Dict = None,
        include_reasoning: bool = True
    ) -> Dict:
        """
        Generate one product launch record with LLM.
        
        Args:
            context: Optional constraints (e.g., {'protein_type': 'beef'})
            include_reasoning: Include reasoning field in output
            
        Returns:
            Dictionary with product data
        """
        prompt = get_single_product_prompt(context)
        
        try:
            response = self._call_llm(prompt)
            product = json.loads(response)
            
            if not include_reasoning and 'reasoning' in product:
                del product['reasoning']
                
            return product
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response[:200]}...")
            raise
        except Exception as e:
            print(f"Error generating product: {e}")
            raise
    
    def generate_batch_with_llm(
        self, 
        batch_size: int = 20,
        include_reasoning: bool = False
    ) -> List[Dict]:
        """
        Generate multiple products in one API call.
        
        Args:
            batch_size: Number of products to generate
            include_reasoning: Include reasoning field in output
            
        Returns:
            List of product dictionaries
        """
        prompt = get_batch_product_prompt(batch_size)
        
        try:
            response = self._call_llm(prompt)
            
            # Handle both array and object responses
            try:
                products = json.loads(response)
                if isinstance(products, dict) and 'products' in products:
                    products = products['products']
                elif not isinstance(products, list):
                    raise ValueError("Response is not a list or dict with 'products' key")
            except json.JSONDecodeError:
                # Try to extract JSON array from response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    products = json.loads(json_match.group())
                else:
                    raise
            
            if not include_reasoning:
                for product in products:
                    if 'reasoning' in product:
                        del product['reasoning']
            
            return products
            
        except Exception as e:
            print(f"Error generating batch: {e}")
            print(f"Response preview: {response[:200] if response else 'None'}...")
            raise
    
    def estimate_cost(self, n_products: int, batch_size: int = 20) -> Dict:
        """
        Estimate API costs for generating products.
        
        Args:
            n_products: Total number of products to generate
            batch_size: Products per API call
            
        Returns:
            Dictionary with cost estimates
        """
        n_batches = (n_products + batch_size - 1) // batch_size
        
        # Rough token estimates
        tokens_per_batch = 3000  # ~1500 input + ~1500 output
        total_tokens = n_batches * tokens_per_batch
        
        # Cost estimates (as of 2024, subject to change)
        if self.provider == 'openai':
            if 'gpt-4' in self.model:
                cost_per_1k_tokens = 0.03  # Average of input/output
            else:
                cost_per_1k_tokens = 0.002  # GPT-3.5
        else:  # Anthropic
            cost_per_1k_tokens = 0.015  # Average for Claude
        
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            'n_products': n_products,
            'n_batches': n_batches,
            'batch_size': batch_size,
            'estimated_tokens': total_tokens,
            'estimated_cost_usd': round(estimated_cost, 2),
            'provider': self.provider,
            'model': self.model
        }


def generate_npd_dataset_with_llm(
    n_products: int = 300,
    batch_size: int = 20,
    provider: str = None,
    output_path: str = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Generate full NPD dataset with LLM and progress tracking.
    
    Args:
        n_products: Total number of products to generate
        batch_size: Products per API call (recommended: 10-20)
        provider: 'openai' or 'anthropic'
        output_path: Optional path to save CSV
        show_progress: Print progress updates
        
    Returns:
        DataFrame with generated NPD data
    """
    generator = LLMDataGenerator(provider=provider)
    
    # Show cost estimate
    if show_progress:
        cost_info = generator.estimate_cost(n_products, batch_size)
        print(f"\n{'='*60}")
        print(f"LLM Data Generation Plan")
        print(f"{'='*60}")
        print(f"Provider: {cost_info['provider']} ({cost_info['model']})")
        print(f"Total products: {cost_info['n_products']}")
        print(f"Batch size: {cost_info['batch_size']}")
        print(f"API calls needed: {cost_info['n_batches']}")
        print(f"Estimated tokens: {cost_info['estimated_tokens']:,}")
        print(f"Estimated cost: ${cost_info['estimated_cost_usd']:.2f} USD")
        print(f"{'='*60}\n")
        
        response = input("Proceed with generation? (y/n): ")
        if response.lower() != 'y':
            print("Generation cancelled.")
            return None
    
    all_products = []
    n_batches = (n_products + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        current_batch_size = min(batch_size, n_products - len(all_products))
        
        if show_progress:
            print(f"Generating batch {batch_idx + 1}/{n_batches} ({current_batch_size} products)...", end=" ")
        
        try:
            batch = generator.generate_batch_with_llm(
                batch_size=current_batch_size,
                include_reasoning=False
            )
            all_products.extend(batch)
            
            if show_progress:
                print(f"✓ ({len(all_products)}/{n_products} total)")
            
            # Rate limiting
            if batch_idx < n_batches - 1:
                time.sleep(1)
                
        except Exception as e:
            print(f"\n❌ Error in batch {batch_idx + 1}: {e}")
            print("Continuing with next batch...")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_products)
    
    # Add product IDs
    df.insert(0, 'product_id', [f'LLM_{i:03d}' for i in range(len(df))])
    
    if show_progress:
        print(f"\n{'='*60}")
        print(f"Generation Complete")
        print(f"{'='*60}")
        print(f"Products generated: {len(df)}")
        print(f"Success rate: {df['success'].mean():.1%}")
        print(f"Average revenue (successful): £{df[df['success']==1]['first_year_revenue_gbp'].mean():,.0f}")
        print(f"Average revenue (failed): £{df[df['success']==0]['first_year_revenue_gbp'].mean():,.0f}")
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        if show_progress:
            print(f"\n✅ Data saved to: {output_path}")
    
    return df


def validate_llm_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate generated LLM data for consistency and realism.
    
    Args:
        df: DataFrame with LLM-generated data
        verbose: Print validation details
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Required columns
    required_cols = [
        'protein_type', 'preparation', 'price_point', 'portion_size_g',
        'shelf_life_days', 'launch_quarter', 'target_channel',
        'competitive_products', 'trend_alignment', 'marketing_spend_gbp',
        'has_sustainability_claim', 'has_origin_story', 'packaging_innovation',
        'development_time_months', 'testing_iterations', 'consultant_involved',
        'success', 'first_year_revenue_gbp'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Validate categorical values
    for col, valid_values in VALIDATION_RULES.items():
        if col not in df.columns:
            continue
            
        if isinstance(valid_values, list):
            invalid = df[~df[col].isin(valid_values)]
            if len(invalid) > 0:
                unique_invalid = invalid[col].unique()
                issues.append(f"{col}: {len(invalid)} rows with invalid values {unique_invalid}")
    
    # Validate numeric ranges
    numeric_ranges = {
        'portion_size_g': (100, 1000),
        'shelf_life_days': (3, 30),
        'competitive_products': (0, 20),
        'trend_alignment': (0.0, 1.0),
        'marketing_spend_gbp': (1000, 75000),
        'development_time_months': (2, 24),
        'testing_iterations': (1, 10)
    }
    
    for col, (min_val, max_val) in numeric_ranges.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
        if len(out_of_range) > 0:
            issues.append(f"{col}: {len(out_of_range)} rows out of range [{min_val}, {max_val}]")
    
    # Check success rate
    success_rate = df['success'].mean()
    if success_rate < 0.35 or success_rate > 0.65:
        issues.append(f"Success rate {success_rate:.1%} outside realistic range (35-65%)")
    
    # Check for logical correlations
    if 'success' in df.columns and 'first_year_revenue_gbp' in df.columns:
        avg_success_revenue = df[df['success'] == 1]['first_year_revenue_gbp'].mean()
        avg_fail_revenue = df[df['success'] == 0]['first_year_revenue_gbp'].mean()
        if avg_success_revenue <= avg_fail_revenue:
            issues.append("Successful products should have higher revenue than failed products")
    
    is_valid = len(issues) == 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Data Validation Report")
        print(f"{'='*60}")
        print(f"Total rows: {len(df)}")
        print(f"Status: {'✅ VALID' if is_valid else '❌ ISSUES FOUND'}")
        if issues:
            print(f"\nIssues ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nAll validation checks passed!")
        print(f"{'='*60}\n")
    
    return is_valid, issues


def merge_with_existing_data(
    llm_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    llm_weight: float = 0.5,
    output_path: str = None
) -> pd.DataFrame:
    """
    Merge LLM-generated data with existing rule-based data.
    
    Args:
        llm_df: DataFrame with LLM-generated products
        existing_df: DataFrame with rule-based products
        llm_weight: Proportion of LLM data (0.0-1.0)
        output_path: Optional path to save merged CSV
        
    Returns:
        Merged DataFrame
    """
    # Calculate sample sizes
    total_llm = len(llm_df)
    total_existing = len(existing_df)
    
    n_llm_samples = int(total_llm * llm_weight)
    n_existing_samples = total_llm - n_llm_samples
    
    # Sample from each dataset
    llm_sample = llm_df.sample(n=min(n_llm_samples, total_llm), random_state=42)
    existing_sample = existing_df.sample(n=min(n_existing_samples, total_existing), random_state=42)
    
    # Combine
    merged_df = pd.concat([llm_sample, existing_sample], ignore_index=True)
    
    # Reassign product IDs
    merged_df['product_id'] = [f'NPD_{i:03d}' for i in range(len(merged_df))]
    
    # Shuffle
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"Data Merge Complete")
    print(f"{'='*60}")
    print(f"LLM products: {len(llm_sample)} ({len(llm_sample)/len(merged_df):.1%})")
    print(f"Rule-based products: {len(existing_sample)} ({len(existing_sample)/len(merged_df):.1%})")
    print(f"Total products: {len(merged_df)}")
    print(f"Success rate: {merged_df['success'].mean():.1%}")
    
    if output_path:
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Merged data saved to: {output_path}")
    
    return merged_df


def main():
    """CLI interface for LLM data generation."""
    parser = argparse.ArgumentParser(
        description='LLM-powered NPD data generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 products using OpenAI
  python data/llm_data_generation.py --provider openai --count 100 --output data/llm_npd_data.csv
  
  # Validate existing data
  python data/llm_data_generation.py --validate data/synthetic_npd_data.csv
  
  # Merge LLM and rule-based data (50/50)
  python data/llm_data_generation.py --merge --llm-weight 0.5
        """
    )
    
    parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic'],
        help='LLM provider (default: from .env LLM_PROVIDER or openai)'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of products to generate (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Products per API call (default: 20)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV path (default: data/llm_npd_data.csv)'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        help='Validate existing CSV file'
    )
    
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge LLM and rule-based data'
    )
    
    parser.add_argument(
        '--llm-weight',
        type=float,
        default=0.5,
        help='Proportion of LLM data when merging (0.0-1.0, default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        print(f"Validating: {args.validate}")
        df = pd.read_csv(args.validate)
        validate_llm_data(df, verbose=True)
        return
    
    # Merge mode
    if args.merge:
        print("Merging LLM and rule-based data...")
        
        # Load data
        llm_path = 'data/llm_npd_data.csv'
        existing_path = 'data/synthetic_npd_data.csv'
        
        if not os.path.exists(llm_path):
            print(f"❌ Error: {llm_path} not found. Generate LLM data first.")
            return
        
        if not os.path.exists(existing_path):
            print(f"❌ Error: {existing_path} not found. Generate rule-based data first.")
            return
        
        llm_df = pd.read_csv(llm_path)
        existing_df = pd.read_csv(existing_path)
        
        output = args.output or 'data/merged_npd_data.csv'
        merge_with_existing_data(llm_df, existing_df, args.llm_weight, output)
        return
    
    # Generation mode
    output_path = args.output or 'data/llm_npd_data.csv'
    
    print(f"Generating {args.count} products using {args.provider or 'default provider'}...")
    
    try:
        df = generate_npd_dataset_with_llm(
            n_products=args.count,
            batch_size=args.batch_size,
            provider=args.provider,
            output_path=output_path,
            show_progress=True
        )
        
        if df is not None:
            # Validate generated data
            print("\nValidating generated data...")
            validate_llm_data(df, verbose=True)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure API key is set in .env file or environment variables")
        print("2. Check internet connection")
        print("3. Verify API key is valid and has credits")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
