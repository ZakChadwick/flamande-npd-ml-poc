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