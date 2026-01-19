"""
Model Evaluation and Metrics

Comprehensive evaluation metrics and visualizations for NPD prediction model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import os


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive classification metrics
    
    Args: 
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
    }
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix heatmap
    
    Args: 
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Failure', 'Success'],
                yticklabels=['Failure', 'Success'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot precision-recall curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Precision-recall curve saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, save_path=None):
    """
    Plot calibration curve to assess probability calibration
    
    Args: 
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        save_path: Path to save plot (optional)
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
             linewidth=2, label='Model Calibration')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Calibration curve saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=15, save_path=None):
    """
    Plot feature importance bar chart
    
    Args: 
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to display
        save_path: Path to save plot (optional)
    """
    # Create DataFrame and sort
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_df)), feature_df['importance'], color='steelblue')
    plt.yticks(range(len(feature_df)), feature_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def business_impact_analysis(y_true, y_pred, y_pred_proba, 
                             avg_dev_cost=30000, 
                             avg_success_revenue=90000,
                             avg_failure_revenue=20000):
    """
    Calculate business impact of model predictions
    
    Args: 
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        avg_dev_cost: Average product development cost
        avg_success_revenue: Average revenue from successful product
        avg_failure_revenue:  Average revenue from failed product
        
    Returns:
        dict:  Business impact metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Without model (launch all products)
    total_products = len(y_true)
    actual_successes = sum(y_true)
    actual_failures = total_products - actual_successes
    
    baseline_cost = total_products * avg_dev_cost
    baseline_revenue = (actual_successes * avg_success_revenue + 
                       actual_failures * avg_failure_revenue)
    baseline_profit = baseline_revenue - baseline_cost
    
    # With model (only launch predicted successes)
    predicted_launches = sum(y_pred)
    model_cost = predicted_launches * avg_dev_cost
    model_revenue = (tp * avg_success_revenue + fp * avg_failure_revenue)
    model_profit = model_revenue - model_cost
    
    # Savings from avoided failures
    avoided_failures = tn  # True negatives - correctly avoided failures
    savings_from_avoided = avoided_failures * avg_dev_cost
    
    # Lost opportunities (false negatives)
    missed_opportunities = fn
    lost_revenue = missed_opportunities * (avg_success_revenue - avg_dev_cost)
    
    impact = {
        'baseline_profit': baseline_profit,
        'model_profit': model_profit,
        'profit_improvement': model_profit - baseline_profit,
        'savings_from_avoided_failures': savings_from_avoided,
        'lost_opportunity_cost': lost_revenue,
        'net_benefit': (model_profit - baseline_profit),
        'roi_improvement_pct': ((model_profit - baseline_profit) / abs(baseline_profit) * 100) 
                                if baseline_profit != 0 else 0,
        'predicted_launches': predicted_launches,
        'avoided_failures': avoided_failures,
        'missed_opportunities': missed_opportunities
    }
    
    return impact


def generate_evaluation_report(y_true, y_pred, y_pred_proba, feature_names=None, 
                               importances=None, output_dir='evaluation_results'):
    """
    Generate comprehensive evaluation report with plots and metrics
    
    Args: 
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        feature_names: List of feature names (optional)
        importances: Feature importance values (optional)
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("GENERATING EVALUATION REPORT")
    print("="*60)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print("\n📊 Classification Metrics:")
    print(f"  Accuracy:   {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    
    # Generate plots
    print("\n📈 Generating plots...")
    plot_confusion_matrix(y_true, y_pred, 
                         save_path=f'{output_dir}/confusion_matrix. png')
    plot_roc_curve(y_true, y_pred_proba, 
                  save_path=f'{output_dir}/roc_curve.png')
    plot_precision_recall_curve(y_true, y_pred_proba, 
                               save_path=f'{output_dir}/precision_recall_curve.png')
    plot_calibration_curve(y_true, y_pred_proba, 
                          save_path=f'{output_dir}/calibration_curve.png')
    
    if feature_names and importances:
        plot_feature_importance(feature_names, importances, 
                              save_path=f'{output_dir}/feature_importance.png')
    
    # Business impact
    print("\n💼 Business Impact Analysis:")
    impact = business_impact_analysis(y_true, y_pred, y_pred_proba)
    print(f"  Baseline Profit:  £{impact['baseline_profit']: ,}")
    print(f"  Model-Guided Profit: £{impact['model_profit']:,}")
    print(f"  Profit Improvement: £{impact['profit_improvement']:,}")
    print(f"  ROI Improvement: {impact['roi_improvement_pct']:.1f}%")
    print(f"  Avoided Failures: {impact['avoided_failures']}")
    print(f"  Savings from Avoided Failures: £{impact['savings_from_avoided_failures']:,}")
    print(f"  Missed Opportunities: {impact['missed_opportunities']}")
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
    
    impact_df = pd.DataFrame([impact])
    impact_df.to_csv(f'{output_dir}/business_impact.csv', index=False)
    
    print(f"\n✅ Evaluation report saved to {output_dir}/")


def main():
    """Demo evaluation functionality"""
    from model import NPDPredictor
    
    print("="*60)
    print("Model Evaluation Demo")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/synthetic_npd_data.csv')
    
    # Load model
    predictor = NPDPredictor. load('models/npd_predictor.pkl')
    
    # Prepare features
    X, feature_names = predictor.prepare_features(df)
    y = df['success']
    
    # Get predictions
    y_pred_proba = predictor.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Get feature importance
    importance_df = predictor.get_feature_importance()
    
    # Generate report
    generate_evaluation_report(
        y, y_pred, y_pred_proba,
        feature_names=importance_df['feature'].values,
        importances=importance_df['importance'].values
    )


if __name__ == "__main__":
    main()