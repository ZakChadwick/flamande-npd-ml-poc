# User Guide: NPD Success Predictor

## Table of Contents
1. [Getting Started](#getting-started)
2. [Using the Web Interface](#using-the-web-interface)
3. [Python API Usage](#python-api-usage)
4. [Interpreting Results](#interpreting-results)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/ZakChadwick/flamande-npd-ml-poc.git
cd flamande-npd-ml-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Initial Setup

```bash
# Generate synthetic training data
python data/data_generation.py

# Train the model
python src/model. py
```

This will create:
- `data/synthetic_npd_data.csv` - Training dataset
- `models/npd_predictor.pkl` - Trained model

---

## Using the Web Interface

### Launch Application

```bash
streamlit run streamlit_app/app.py
```

The app opens at `http://localhost:8501`

### Interface Overview

#### 1. Sidebar:  Product Configuration

**Product Attributes**
- **Protein Type**: Primary protein (beef, pork, lamb, chicken, mixed)
- **Preparation**: Processing style (raw, marinated, seasoned, ready-to-cook, cooked)
- **Price Point**: Market positioning (economy, mid-range, premium)
- **Portion Size**: Product weight in grams
- **Shelf Life**: Days until expiration

**Market Context**
- **Target Channel**: Primary sales channel
- **Launch Quarter**: Planned launch timing
- **Competitive Products**: Number of direct competitors
- **Trend Alignment**: How well concept aligns with current trends (0-100)

**Marketing Strategy**
- **Marketing Budget**: Allocated spend in GBP
- **Sustainability Claim**: Environmental/ethical positioning
- **Origin Story**: Authenticity narrative
- **Packaging Innovation**: Novel packaging features

**Development Info**
- **Development Time**: Months from concept to launch
- **Testing Iterations**: Number of refinement cycles
- **Consultant Involved**: Professional NPD support

#### 2. Prediction Tab

Shows: 
- **Success Probability**: 0-100% likelihood of success
- **Estimated Revenue**: Projected first-year sales
- **Risk Level**: Low/Medium/High categorization
- **Gauge Chart**: Visual probability display
- **Interpretation**:  Actionable recommendation

#### 3. Optimization Tab

Features:
- **Top Recommendations**: Ranked improvements
- **Impact Analysis**: Revenue and probability lift
- **Multi-Factor Optimization**: Combined modifications
- **Investment ROI**: Cost-benefit for changes

#### 4. Insights Tab

Provides:
- **Success Factors**: Industry-wide drivers
- **Market Trends**: Temporal patterns
- **Channel Performance**: By distribution type
- **Best Practices**: Data-backed guidance

---

## Python API Usage

### Basic Prediction

```python
from src.model import NPDPredictor

# Load trained model
predictor = NPDPredictor. load('models/npd_predictor. pkl')

# Define product concept
concept = {
    'protein_type': 'beef',
    'preparation':  'marinated',
    'price_point': 'premium',
    'portion_size_g': 300,
    'shelf_life_days': 14,
    'launch_quarter': 'Q2',
    'target_channel': 'food_service',
    'competitive_products': 5,
    'trend_alignment':  0.75,
    'marketing_spend_gbp': 25000,
    'has_sustainability_claim': 1,
    'has_origin_story': 1,
    'packaging_innovation': 0,
    'development_time_months': 6,
    'testing_iterations': 3,
    'consultant_involved': 1
}

# Predict success probability
probability = predictor.predict_single(concept)
print(f"Success Probability: {probability:.1%}")
```

### Product Optimization

```python
from src.optimizer import NPDOptimizer

# Initialize optimizer
optimizer = NPDOptimizer(predictor)

# Get recommendations
recommendations = optimizer.optimize_product_concept(concept, top_n=5)

for rec in recommendations:
    print(f"{rec['change']}:  +{rec['lift']*100:.1f}%")
```

### Batch Processing

```python
import pandas as pd

# Load multiple concepts
concepts_df = pd.read_csv('my_concepts.csv')

# Prepare features
X, _ = predictor.prepare_features(concepts_df)

# Predict all
probabilities = predictor.predict(X)

# Add to dataframe
concepts_df['success_probability'] = probabilities
concepts_df['risk_level'] = pd.cut(
    probabilities,
    bins=[0, 0.45, 0.65, 1.0],
    labels=['High', 'Medium', 'Low']
)

# Save results
concepts_df. to_csv('predictions.csv', index=False)
```

### Scenario Analysis

```python
scenarios = [
    {'preparation': 'marinated', 'has_sustainability_claim': 1},
    {'price_point': 'premium', 'target_channel': 'food_service'},
    {'preparation': 'ready-to-cook', 'packaging_innovation': 1}
]

results = optimizer.scenario_analysis(concept, scenarios)
print(results)
```

---

## Interpreting Results

### Success Probability Ranges

| Range | Risk Level | Interpretation | Action |
|-------|------------|----------------|--------|
| 0-45% | 🔴 High | Significant challenges | Major rework or abandon |
| 45-65% | 🟡 Medium | Moderate potential | Review optimizations |
| 65-100% | 🟢 Low | Strong potential | Proceed with confidence |

### Understanding Recommendations

**Lift**:  Absolute percentage point increase  
**Relative Lift**: Proportional improvement

Example:
- Baseline: 50%
- New:  60%
- Lift: +10 percentage points
- Relative Lift: +20% (10/50)

### Feature Importance

Model decisions are based on:
1. **Preparation** (30%): Convenience matters
2. **Channel** (25%): Distribution alignment
3. **Price** (20%): Value proposition
4. **Claims** (15%): Differentiation
5. **Shelf Life** (10%): Operational feasibility

---

## Best Practices

### Data Input

✅ **Do:**
- Use realistic, market-researched values
- Be honest about competitive landscape
- Update trend alignment regularly
- Include all planned marketing investments

❌ **Don't:**
- Inflate trend scores without evidence
- Underestimate competition
- Ignore shelf life constraints
- Skip sustainability/origin opportunities

### Using Predictions

✅ **Do:**
- Use early in concept phase
- Test multiple variations
- Combine with qualitative insights
- Re-evaluate after market changes

❌ **Don't:**
- Use as sole decision factor
- Ignore domain expertise
- Skip pilot/testing phases
- Apply to radically different categories

### Optimization

✅ **Do:**
- Prioritize high-impact, low-cost changes
- Consider feasibility of recommendations
- Test combined optimizations
- Validate with target customers

❌ **Don't:**
- Make changes that contradict brand
- Ignore manufacturing constraints
- Optimize beyond realistic capabilities
- Sacrifice quality for predicted success

---

## Troubleshooting

### Model Not Found Error

```
⚠️ Model not found.  Please train the model first
```

**Solution:**
```bash
python src/model.py
```

### Data Generation Issues

```
FileNotFoundError: data/synthetic_npd_data. csv
```

**Solution:**
```bash
python data/data_generation.py
```

### Dependency Errors

```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Streamlit Won't Start

**Check Python version:**
```bash
python --version  # Should be 3.9+
```

**Reinstall Streamlit:**
```bash
pip install --upgrade streamlit
```

### Predictions Seem Unrealistic

**Check input data:**
- Ensure all values are in correct ranges
- Verify categorical values match expected options
- Confirm numeric fields aren't swapped

**Retrain model:**
```bash
python src/model.py
```

### Performance Issues

**Large batch predictions:**
- Process in chunks of 1000 records
- Use multiprocessing for parallel predictions

```python
from joblib import Parallel, delayed

def predict_batch(concepts_chunk):
    return predictor.predict(concepts_chunk)

# Parallel processing
results = Parallel(n_jobs=4)(
    delayed(predict_batch)(chunk) 
    for chunk in np.array_split(X, 4)
)
```

---

## Support

### Additional Help

- **Documentation**: See `docs/Technical_Documentation.md`
- **Examples**: Check `notebooks/` directory
- **Issues**: GitHub Issues tab
- **Email**: support@example.com

### Feedback

We welcome feedback to improve the tool: 
- Feature requests
- Bug reports
- Use case suggestions
- Accuracy improvements

**Submit via:** GitHub Issues or email

---

**Last Updated:** January 2026  
**Version:** 1.0.0