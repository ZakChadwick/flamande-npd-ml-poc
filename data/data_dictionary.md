# Data Dictionary

## NPD Dataset Features

### Product Characteristics

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| `product_id` | string | Unique product identifier | NPD_000 to NPD_299 |
| `protein_type` | categorical | Primary protein source | beef, pork, lamb, chicken, mixed |
| `preparation` | categorical | Preparation/processing style | raw, marinated, seasoned, ready-to-cook, cooked |
| `price_point` | categorical | Price positioning | economy, mid-range, premium |
| `portion_size_g` | integer | Product portion size in grams | 200, 250, 300, 400, 500 |
| `shelf_life_days` | integer | Shelf life in days | 3-20 |

### Market Context

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| `launch_quarter` | categorical | Quarter of product launch | Q1, Q2, Q3, Q4 |
| `target_channel` | categorical | Primary sales channel | food_service, retail_butcher, supermarket, direct |
| `competitive_products` | integer | Number of competing products | 0-14 |
| `trend_alignment` | float | Alignment with current trends (0-1 score) | 0.0-1.0 |

### Marketing Factors

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| `marketing_spend_gbp` | integer | Marketing budget in GBP | 1,000-50,000 |
| `has_sustainability_claim` | binary | Product has sustainability claim | 0=No, 1=Yes |
| `has_origin_story` | binary | Product has origin/authenticity story | 0=No, 1=Yes |
| `packaging_innovation` | binary | Innovative packaging features | 0=No, 1=Yes |

### Development Factors

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| `development_time_months` | integer | Time from concept to launch | 2-17 months |
| `testing_iterations` | integer | Number of product testing cycles | 1-7 |
| `consultant_involved` | binary | NPD consultant involvement | 0=No, 1=Yes |

### Outcomes

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| `success` | binary | Product success (target variable) | 0=Failed, 1=Successful |
| `first_year_revenue_gbp` | integer | First year revenue in GBP | Variable |
| `success_score` | float | Internal success probability score | 0.0-1.0 (for reference only) |

---

## Success Definition

A product is considered **successful** (`success=1`) if it:
- Achieves profitability within 12 months
- Maintains consistent sales velocity
- Receives positive customer feedback
- Gets reordered by distribution partners

**Failure** (`success=0`) indicates:
- Discontinued within 2 years
- Negative ROI
- Poor market reception
- Distribution rejection

---

## Feature Engineering Notes

### Categorical Encoding
All categorical features are label-encoded for model training:
- Alphabetically sorted
- Consistent encoding across train/test sets
- Stored encoders for production use

### Feature Interactions
Key interaction effects:
- `price_point` × `target_channel` (premium works in food service)
- `preparation` × `protein_type` (marinated lamb performs well)
- `sustainability_claim` × `price_point` (premium buyers value sustainability)

### Missing Data
This synthetic dataset has no missing values.  In production: 
- `competitive_products`: Impute with channel median
- `trend_alignment`: Research or set to 0.5 (neutral)
- `marketing_spend_gbp`: Impute with price point median

---

## Data Generation Logic

The synthetic data is generated to reflect real-world patterns:

### Success Drivers (Positive Impact)
1. **Premium + Food Service** (+25%)
2. **Marinated/Ready-to-Cook** (+20%)
3. **High Trend Alignment** (+15%)
4. **Sustainability Claims** (+12%)
5. **Marketing Investment** (up to +12%)
6. **Consultant Involved** (+10%)

### Success Inhibitors (Negative Impact)
1. **High Competition** (variable -)
2. **Raw Preparation** (-10%)
3. **Supermarket Channel** (-5%)
4. **Short Shelf Life** (variable -)

### Realistic Noise
- ±12% random variation added to success scores
- Simulates real-world unpredictability
- Creates ~45-50% baseline success rate

---

## Usage Examples

### Load Data
```python
import pandas as pd

df = pd.read_csv('data/synthetic_npd_data. csv')
```

### Filter Successful Products
```python
successful = df[df['success'] == 1]
```

### Analyze by Channel
```python
channel_success = df.groupby('target_channel')['success'].mean()
```

### Feature Correlation
```python
correlation_matrix = df.corr()['success'].sort_values(ascending=False)
```

---

## LLM-Powered Data Generation

### Overview

In addition to rule-based data generation (`data_generation.py`), this project supports **LLM-powered data generation** using generative AI models (OpenAI GPT-4 or Anthropic Claude). This approach leverages the LLM's knowledge of real-world food industry patterns to create more realistic and nuanced synthetic data.

### How It Works

The LLM data generator (`llm_data_generation.py`) uses carefully crafted prompts to instruct the AI model to:

1. **Act as a UK food industry expert** with 20 years of butchery and food service experience
2. **Generate realistic product launches** with complete feature sets matching the schema
3. **Apply industry knowledge** about market dynamics, trends, and success factors
4. **Maintain realistic failure rates** (~40-50% as per industry standards)
5. **Create logical correlations** between features (e.g., premium products with higher marketing spend)

### Key Differences: Rule-Based vs LLM-Generated Data

| Aspect | Rule-Based | LLM-Generated |
|--------|------------|---------------|
| **Approach** | Hardcoded probability distributions | AI-inferred industry patterns |
| **Correlations** | Explicitly programmed | Learned from training data |
| **Diversity** | Limited by predefined rules | More varied and nuanced |
| **Realism** | Based on developer assumptions | Based on LLM's industry knowledge |
| **Cost** | Free | API costs (~$0.50-2.00 per 1000 products) |
| **Speed** | Instant | ~1-2 seconds per product |
| **Reproducibility** | Fully deterministic | Stochastic (temperature-based) |

### Using LLM Data Generation

#### Prerequisites

1. **Install dependencies:**
```bash
pip install openai>=1.0.0 anthropic>=0.18.0 python-dotenv>=1.0.0 tenacity>=8.2.0
```

2. **Set up API keys:**
```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=your-key-here
# OR
ANTHROPIC_API_KEY=your-key-here
LLM_PROVIDER=openai  # or anthropic
```

#### Generate LLM Data

```bash
# Generate 100 products using OpenAI
python data/llm_data_generation.py --provider openai --count 100 --output data/llm_npd_data.csv

# Generate using Anthropic Claude
python data/llm_data_generation.py --provider anthropic --count 200

# Validate existing data
python data/llm_data_generation.py --validate data/synthetic_npd_data.csv

# Merge LLM and rule-based data (50% each)
python data/llm_data_generation.py --merge --llm-weight 0.5
```

#### Programmatic Usage

```python
from data.llm_data_generation import LLMDataGenerator, generate_npd_dataset_with_llm

# Initialize generator
generator = LLMDataGenerator(provider='openai', temperature=0.8)

# Generate single product
product = generator.generate_single_product_with_llm(
    context={'protein_type': 'beef'},
    include_reasoning=True
)

# Generate batch
products = generator.generate_batch_with_llm(batch_size=20)

# Generate full dataset
df = generate_npd_dataset_with_llm(
    n_products=300,
    batch_size=20,
    provider='openai',
    output_path='data/llm_npd_data.csv'
)
```

### Best Practices

1. **Start with small batches** (20-50 products) to validate output quality before generating large datasets
2. **Use batch generation** (`batch_size=10-20`) to reduce API costs and improve throughput
3. **Monitor costs** - the tool provides cost estimates before generation
4. **Validate data** - always run validation after generation to check for anomalies
5. **Mix data sources** - combine LLM and rule-based data for best results (e.g., 50/50 split)
6. **Set appropriate temperature** - higher (0.8-1.0) for diversity, lower (0.3-0.5) for consistency
7. **Use reasoning field** during development to understand LLM's decision-making

### Data Validation

The `validate_llm_data()` function checks for:

- **Schema compliance** - all required columns present
- **Valid categorical values** - protein types, preparation styles, etc.
- **Numeric ranges** - portion sizes, shelf life, marketing spend within bounds
- **Realistic failure rate** - 35-65% success rate
- **Logical correlations** - successful products have higher revenue
- **Consistent pricing** - premium products generate appropriate revenue

### Known Limitations and Biases

1. **LLM training cutoff** - Models may not reflect very recent market trends
2. **Geographic bias** - LLM knowledge may be stronger for US/global markets than UK-specific patterns
3. **Optimism bias** - LLMs may slightly overestimate success rates unless prompted carefully
4. **Cost sensitivity** - Large datasets (1000+ products) can become expensive ($2-10 USD)
5. **API reliability** - Rate limits and outages can interrupt large generations
6. **Non-deterministic** - Same prompts may yield different results (use temperature=0 for consistency)

### Recommended Approach

For production use, we recommend a **hybrid approach**:

1. **Generate 50% rule-based data** - provides baseline consistency and zero cost
2. **Generate 50% LLM data** - adds realism and pattern diversity
3. **Validate merged dataset** - ensure combined data maintains quality standards
4. **Re-train model** - compare performance on pure vs. hybrid datasets

```bash
# Generate rule-based data
python data/data_generation.py

# Generate LLM data
python data/llm_data_generation.py --provider openai --count 300

# Merge datasets
python data/llm_data_generation.py --merge --llm-weight 0.5 --output data/merged_npd_data.csv
```

This approach balances **cost, speed, and realism** while reducing the risk of LLM biases dominating the training data.

---

## Cost Estimation

### LLM Generation Costs (Approximate)

| Products | API Calls | Tokens | OpenAI (GPT-4) | Anthropic (Claude) |
|----------|-----------|--------|----------------|-------------------|
| 100 | 5-10 | ~15k | $0.45 | $0.23 |
| 300 | 15-20 | ~45k | $1.35 | $0.68 |
| 1000 | 50-100 | ~150k | $4.50 | $2.25 |

**Note:** Costs vary based on model version, prompt length, and token pricing. Always check the cost estimate shown before confirming generation.