"""
LLM Prompt Templates for NPD Data Generation

Contains reusable prompts for generating synthetic product launch data
using LLMs (OpenAI GPT-4 or Anthropic Claude).
"""

SYSTEM_PROMPT = """You are a UK food industry expert with 20 years of experience in butchery 
and food service product development. You have deep knowledge of:
- UK meat and protein market dynamics
- Consumer trends in food service and retail
- Product development success factors
- Marketing strategies for food products
- Distribution channel requirements

Your role is to generate realistic new product development (NPD) data based on 
actual industry patterns, ensuring a realistic failure rate of 40-50% as seen in 
the real food industry."""

FEW_SHOT_EXAMPLES = [
    {
        "description": "Example of a successful premium product",
        "product": {
            "protein_type": "beef",
            "preparation": "marinated",
            "price_point": "premium",
            "portion_size_g": 400,
            "shelf_life_days": 14,
            "launch_quarter": "Q2",
            "target_channel": "food_service",
            "competitive_products": 3,
            "trend_alignment": 0.85,
            "marketing_spend_gbp": 45000,
            "has_sustainability_claim": 1,
            "has_origin_story": 1,
            "packaging_innovation": 1,
            "development_time_months": 12,
            "testing_iterations": 6,
            "consultant_involved": 1,
            "success": 1,
            "first_year_revenue_gbp": 185000,
            "reasoning": "Premium marinated beef launched in Q2 for BBQ season with strong sustainability credentials and origin story. High marketing spend and consultant involvement led to excellent food service adoption. Low competition in premium segment."
        }
    },
    {
        "description": "Example of a failed economy product",
        "product": {
            "protein_type": "chicken",
            "preparation": "raw",
            "price_point": "economy",
            "portion_size_g": 200,
            "shelf_life_days": 5,
            "launch_quarter": "Q1",
            "target_channel": "supermarket",
            "competitive_products": 18,
            "trend_alignment": 0.25,
            "marketing_spend_gbp": 5000,
            "has_sustainability_claim": 0,
            "has_origin_story": 0,
            "packaging_innovation": 0,
            "development_time_months": 3,
            "testing_iterations": 2,
            "consultant_involved": 0,
            "success": 0,
            "first_year_revenue_gbp": 15000,
            "reasoning": "Raw chicken in crowded economy supermarket segment with minimal differentiation. Short shelf life limited distribution. Low marketing spend and no unique claims failed to break through high competition. Poor trend alignment with consumer shift toward convenience."
        }
    }
]


def get_single_product_prompt(context: dict = None) -> str:
    """
    Generate a prompt for creating one realistic product launch.
    
    Args:
        context: Optional dict with constraints (e.g., specific protein_type)
        
    Returns:
        Formatted prompt string
    """
    base_prompt = """Generate ONE realistic new meat product launch for the UK market with complete details.
Consider real UK market conditions, consumer trends, and success factors.

IMPORTANT: Ensure a realistic failure rate - about 40-50% of products should have success=0.

Required fields (use EXACT values from options):

Product Characteristics:
- protein_type: [beef, pork, lamb, chicken, mixed]
- preparation: [raw, marinated, seasoned, ready-to-cook, cooked]
- price_point: [economy, mid-range, premium]
- portion_size_g: [integer between 100-1000]
- shelf_life_days: [integer between 3-30]

Market Context:
- launch_quarter: [Q1, Q2, Q3, Q4]
- target_channel: [food_service, retail_butcher, supermarket, direct]
- competitive_products: [integer 0-20, number of similar products in market]
- trend_alignment: [float 0.0-1.0, how well it matches current trends]

Marketing Factors:
- marketing_spend_gbp: [integer 1000-75000]
- has_sustainability_claim: [0 or 1]
- has_origin_story: [0 or 1]
- packaging_innovation: [0 or 1]

Development Factors:
- development_time_months: [integer 2-24]
- testing_iterations: [integer 1-10]
- consultant_involved: [0 or 1]

Outcomes:
- success: [0 or 1] - Did this product succeed after 18 months?
- first_year_revenue_gbp: [integer, realistic based on success and price point]

Also provide:
- reasoning: 2-3 sentences explaining WHY this product succeeded or failed, considering market context, competition, and product attributes.

Return ONLY valid JSON with these exact field names. No additional text or formatting."""

    if context:
        constraint_text = "\n\nConstraints for this product:\n"
        for key, value in context.items():
            constraint_text += f"- {key}: {value}\n"
        base_prompt += constraint_text

    return base_prompt


def get_batch_product_prompt(batch_size: int = 20) -> str:
    """
    Generate a prompt for creating multiple products in one API call.
    
    Args:
        batch_size: Number of products to generate
        
    Returns:
        Formatted prompt string
    """
    return f"""Generate {batch_size} realistic new meat product launches for the UK market.
Ensure diversity across all dimensions:
- Mix of protein types (beef, pork, lamb, chicken, mixed)
- Various preparation styles and price points
- Different target channels and quarters
- Range of marketing strategies
- REALISTIC failure rate: aim for 40-50% with success=0

Each product should be complete and realistic, considering:
- Market competition and trends
- Logical correlations (e.g., premium products typically have higher marketing spend)
- Seasonal factors (e.g., BBQ season for Q2/Q3)
- Channel requirements (e.g., food service values convenience)

Required fields for EACH product (use EXACT values from options):

Product Characteristics:
- protein_type: [beef, pork, lamb, chicken, mixed]
- preparation: [raw, marinated, seasoned, ready-to-cook, cooked]
- price_point: [economy, mid-range, premium]
- portion_size_g: [integer between 100-1000]
- shelf_life_days: [integer between 3-30]

Market Context:
- launch_quarter: [Q1, Q2, Q3, Q4]
- target_channel: [food_service, retail_butcher, supermarket, direct]
- competitive_products: [integer 0-20]
- trend_alignment: [float 0.0-1.0]

Marketing Factors:
- marketing_spend_gbp: [integer 1000-75000]
- has_sustainability_claim: [0 or 1]
- has_origin_story: [0 or 1]
- packaging_innovation: [0 or 1]

Development Factors:
- development_time_months: [integer 2-24]
- testing_iterations: [integer 1-10]
- consultant_involved: [0 or 1]

Outcomes:
- success: [0 or 1]
- first_year_revenue_gbp: [integer]
- reasoning: 2-3 sentences explaining success/failure

Return a JSON array of {batch_size} products. Each product should be a complete JSON object.
Return ONLY the JSON array, no additional text or formatting."""


def get_validation_prompt(product_data: dict) -> str:
    """
    Generate a prompt for validating generated product data.
    
    Args:
        product_data: Dictionary with product fields
        
    Returns:
        Formatted validation prompt
    """
    return f"""Review this product launch data for realism and consistency:

{product_data}

Check for:
1. Logical correlations (e.g., premium products should have higher marketing spend)
2. Realistic failure patterns (failed products should have clear weaknesses)
3. Market fit (does the product make sense for the target channel?)
4. Trend alignment (does it match with preparation style and price point?)
5. Revenue consistency (does revenue align with success, price point, and channel?)

Respond with:
- is_valid: true/false
- issues: list of any problems found
- suggestions: recommendations to improve realism

Return as JSON."""


VALIDATION_RULES = {
    "protein_type": ["beef", "pork", "lamb", "chicken", "mixed"],
    "preparation": ["raw", "marinated", "seasoned", "ready-to-cook", "cooked"],
    "price_point": ["economy", "mid-range", "premium"],
    "launch_quarter": ["Q1", "Q2", "Q3", "Q4"],
    "target_channel": ["food_service", "retail_butcher", "supermarket", "direct"],
    "portion_size_g": {"min": 100, "max": 1000},
    "shelf_life_days": {"min": 3, "max": 30},
    "competitive_products": {"min": 0, "max": 20},
    "trend_alignment": {"min": 0.0, "max": 1.0},
    "marketing_spend_gbp": {"min": 1000, "max": 75000},
    "has_sustainability_claim": [0, 1],
    "has_origin_story": [0, 1],
    "packaging_innovation": [0, 1],
    "development_time_months": {"min": 2, "max": 24},
    "testing_iterations": {"min": 1, "max": 10},
    "consultant_involved": [0, 1],
    "success": [0, 1]
}
