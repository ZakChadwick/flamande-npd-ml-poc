"""
Synthetic NPD Data Generator using Google Gemini

Generates realistic new product development data for butchery & food service
using Google's Gemini AI for more realistic and varied product concepts.
"""

import pandas as pd
import json
import os
import time
from typing import Optional

# Default API key (can be overridden via environment variable or parameter)
DEFAULT_API_KEY = "AIzaSyDZn0upLXrfOtN62GevseOqJ4WE_NK3ojo"

try:
    import google.generativeai as genai
except ImportError:
    print("Please install the Google Generative AI package:")
    print("  pip install google-generativeai")
    exit(1)


def configure_gemini(api_key: Optional[str] = None):
    """
    Configure the Gemini API client

    Args:
        api_key: Google API key. If None, will use DEFAULT_API_KEY or GOOGLE_API_KEY env var
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY", DEFAULT_API_KEY)

    if not api_key:
        raise ValueError(
            "No API key provided. Either pass api_key parameter or "
            "set GOOGLE_API_KEY environment variable.\n"
            "Get your API key at: https://makersuite.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def generate_product_batch(model, batch_size: int = 10) -> list:
    """
    Generate a batch of NPD product concepts using Gemini

    Args:
        model: Configured Gemini model
        batch_size: Number of products to generate per batch

    Returns:
        List of product dictionaries
    """
    prompt = f"""Generate {batch_size} realistic new product development (NPD) concepts for a UK butchery/meat processing company targeting food service and retail markets.

For each product, provide the following fields in valid JSON format:

- product_name: Creative product name
- product_description: Brief description (1-2 sentences)
- protein_type: One of [beef, pork, lamb, chicken, mixed]
- preparation: One of [raw, marinated, seasoned, ready-to-cook, cooked]
- price_point: One of [economy, mid-range, premium]
- portion_size_g: Integer from [200, 250, 300, 400, 500]
- shelf_life_days: Integer between 3-21
- launch_quarter: One of [Q1, Q2, Q3, Q4]
- target_channel: One of [food_service, retail_butcher, supermarket, direct]
- competitive_products: Integer 0-15 (number of similar products in market)
- trend_alignment: Float 0.0-1.0 (how well it aligns with current food trends)
- marketing_spend_gbp: Integer 1000-50000
- has_sustainability_claim: 0 or 1
- has_origin_story: 0 or 1
- packaging_innovation: 0 or 1
- development_time_months: Integer 2-18
- testing_iterations: Integer 1-8
- consultant_involved: 0 or 1
- success: 0 or 1 (predict based on realistic factors - premium food service products with sustainability claims tend to succeed, raw economy products with high competition tend to fail)
- success_reasoning: Brief explanation of why this product would succeed or fail

Return ONLY a valid JSON array with {batch_size} product objects. No markdown, no explanation, just the JSON array.

Make the products diverse and realistic for the UK meat industry in 2025-2026. Include trendy products (plant-based hybrids, air-fryer ready, sous-vide prepared) and traditional ones."""

    try:
        response = model.generate_content(prompt)

        # Extract JSON from response
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        products = json.loads(response_text)
        return products

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response.text[:500]}...")
        return []
    except Exception as e:
        print(f"Error generating batch: {e}")
        return []


def calculate_revenue(product: dict) -> int:
    """
    Calculate estimated first year revenue based on product attributes

    Args:
        product: Product dictionary

    Returns:
        Estimated first year revenue in GBP
    """
    import random

    if product.get('success', 0) == 1:
        base_revenue = random.gauss(90000, 35000)
    else:
        base_revenue = random.gauss(25000, 15000)

    # Premium products have higher revenue potential
    if product.get('price_point') == 'premium':
        base_revenue *= 1.4
    elif product.get('price_point') == 'economy':
        base_revenue *= 0.8

    return max(0, int(base_revenue))


def generate_npd_data_gemini(
    n_products: int = 100,
    api_key: Optional[str] = None,
    batch_size: int = 10,
    delay_between_batches: float = 1.0
) -> pd.DataFrame:
    """
    Generate NPD dataset using Google Gemini

    Args:
        n_products: Total number of products to generate
        api_key: Google API key (or set GOOGLE_API_KEY env var)
        batch_size: Products per API call (max ~20 recommended)
        delay_between_batches: Seconds to wait between API calls

    Returns:
        DataFrame with generated NPD data
    """
    print("🚀 Initializing Gemini data generator...")
    model = configure_gemini(api_key)

    all_products = []
    n_batches = (n_products + batch_size - 1) // batch_size

    print(f"📦 Generating {n_products} products in {n_batches} batches...")

    for batch_num in range(n_batches):
        remaining = n_products - len(all_products)
        current_batch_size = min(batch_size, remaining)

        print(f"  Batch {batch_num + 1}/{n_batches} ({current_batch_size} products)...", end=" ")

        products = generate_product_batch(model, current_batch_size)

        if products:
            all_products.extend(products)
            print(f"✅ Got {len(products)} products")
        else:
            print("❌ Failed, retrying...")
            time.sleep(2)
            products = generate_product_batch(model, current_batch_size)
            if products:
                all_products.extend(products)
                print(f"  ✅ Retry successful: {len(products)} products")

        # Rate limiting
        if batch_num < n_batches - 1:
            time.sleep(delay_between_batches)

    print(f"\n✅ Generated {len(all_products)} products total")

    # Post-process products
    for i, product in enumerate(all_products):
        product['product_id'] = f'NPD_GEM_{i:03d}'
        product['first_year_revenue_gbp'] = calculate_revenue(product)

        # Calculate success_score based on attributes
        success_score = 0.3
        if product.get('price_point') == 'premium' and product.get('target_channel') == 'food_service':
            success_score += 0.25
        if product.get('preparation') in ['marinated', 'ready-to-cook']:
            success_score += 0.20
        if product.get('has_sustainability_claim') == 1:
            success_score += 0.12
        if product.get('has_origin_story') == 1:
            success_score += 0.08
        success_score -= product.get('competitive_products', 0) / 40
        success_score += product.get('trend_alignment', 0.5) * 0.15
        product['success_score'] = round(min(1, max(0, success_score)), 3)

    # Create DataFrame
    df = pd.DataFrame(all_products)

    # Ensure required columns exist with correct types
    required_columns = [
        'product_id', 'protein_type', 'preparation', 'price_point',
        'portion_size_g', 'shelf_life_days', 'launch_quarter', 'target_channel',
        'competitive_products', 'trend_alignment', 'marketing_spend_gbp',
        'has_sustainability_claim', 'has_origin_story', 'packaging_innovation',
        'development_time_months', 'testing_iterations', 'consultant_involved',
        'success', 'first_year_revenue_gbp', 'success_score'
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    extra_cols = [c for c in df.columns if c not in required_columns]
    df = df[required_columns + extra_cols]

    return df


def main():
    """Generate and save Gemini-powered NPD data"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate NPD data using Google Gemini")
    parser.add_argument("--n", type=int, default=50, help="Number of products to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_npd_data_gemini.csv",
                       help="Output CSV file path")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Products per API call")
    args = parser.parse_args()

    print("="*60)
    print("NPD Data Generation using Google Gemini")
    print("="*60)

    # Generate data
    df = generate_npd_data_gemini(
        n_products=args.n,
        api_key=args.api_key,
        batch_size=args.batch_size
    )

    # Display summary
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Total products: {len(df)}")
    print(f"Success rate: {df['success'].mean():.1%}")

    if 'first_year_revenue_gbp' in df.columns:
        successful = df[df['success'] == 1]['first_year_revenue_gbp']
        failed = df[df['success'] == 0]['first_year_revenue_gbp']
        if len(successful) > 0:
            print(f"Avg revenue (successful): £{successful.mean():,.0f}")
        if len(failed) > 0:
            print(f"Avg revenue (failed): £{failed.mean():,.0f}")

    print(f"\n{'='*60}")
    print("Product Distribution")
    print(f"{'='*60}")
    print("\nProtein Types:")
    print(df['protein_type'].value_counts())
    print("\nPreparation Styles:")
    print(df['preparation'].value_counts())
    print("\nPrice Points:")
    print(df['price_point'].value_counts())
    print("\nTarget Channels:")
    print(df['target_channel'].value_counts())

    # Show sample product names/descriptions if available
    if 'product_name' in df.columns:
        print(f"\n{'='*60}")
        print("Sample Products")
        print(f"{'='*60}")
        for _, row in df.head(5).iterrows():
            print(f"\n📦 {row.get('product_name', 'N/A')}")
            print(f"   {row.get('product_description', 'N/A')}")
            print(f"   {row['protein_type'].title()} | {row['preparation'].title()} | {row['price_point'].title()}")
            print(f"   Success: {'✅' if row['success'] == 1 else '❌'} - {row.get('success_reasoning', 'N/A')[:80]}...")

    # Save to CSV
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Data saved to {args.output}")

    return df


if __name__ == "__main__":
    main()
