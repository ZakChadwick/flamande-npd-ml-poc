"""
NPD Success Prediction Model

Gradient Boosting Classifier for predicting new product development success
in butchery and food service sectors.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)
import joblib
import os


class NPDPredictor:
    """
    NPD Success Prediction Model
    
    Predicts likelihood of new product development success based on
    product attributes, market context, and marketing strategy.
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = [
            'protein_type', 'preparation', 'price_point',
            'launch_quarter', 'target_channel'
        ]
        
    def prepare_features(self, df):
        """
        Encode categorical features and prepare feature matrix
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            X: Feature matrix
            feature_names: List of feature names
        """
        df_encoded = df.copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col + '_encoded'] = self.label_encoders[col].transform(df[col])

        # Define feature columns
        feature_cols = [
            'protein_type_encoded', 'preparation_encoded', 'price_point_encoded',
            'portion_size_g', 'shelf_life_days', 'launch_quarter_encoded',
            'target_channel_encoded', 'competitive_products', 'trend_alignment',
            'marketing_spend_gbp', 'has_sustainability_claim', 'has_origin_story',
            'packaging_innovation', 'development_time_months', 'testing_iterations',
            'consultant_involved'
        ]
        
        self.feature_names = feature_cols
        
        return df_encoded[feature_cols], feature_cols
    
    def train(self, X_train, y_train, n_estimators=200, learning_rate=0.05, max_depth=4):
        """
        Train the gradient boosting model
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum tree depth
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbose=1
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, X):
        """
        Predict success probability
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Success probabilities
        """
        return self.model.predict_proba(X)[:, 1]
    
    def predict_single(self, product_concept):
        """
        Predict success for a single product concept
        
        Args:
            product_concept: dict with product attributes
            
        Returns:
            float: Success probability
        """
        # Convert to DataFrame
        df = pd.DataFrame([product_concept])
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        # Predict
        probability = self.predict(X)[0]
        
        return probability
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            DataFrame: Features ranked by importance
        """
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save(self, filepath='models/npd_predictor.pkl'):
        """Save model and encoders"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/npd_predictor.pkl'):
        """Load model and encoders"""
        predictor = cls()
        model_data = joblib.load(filepath)
        
        predictor.model = model_data['model']
        predictor.label_encoders = model_data['label_encoders']
        predictor.feature_names = model_data['feature_names']
        predictor.categorical_features = model_data['categorical_features']
        
        return predictor


def main():
    """Train and evaluate NPD prediction model"""
    
    print("="*60)
    print("NPD Success Prediction Model Training")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/synthetic_npd_data.csv')
    print(f"Loaded {len(df)} products")
    
    # Initialize predictor
    predictor = NPDPredictor()
    
    # Prepare features
    print("\nPreparing features...")
    X, feature_names = predictor.prepare_features(df)
    y = df['success']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining model...")
    predictor.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\nAccuracy: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"\n{metrics['classification_report']}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    feature_importance = predictor.get_feature_importance()
    print(feature_importance.head(10))

    # Save model
    predictor.save()


if __name__ == "__main__":
    main()