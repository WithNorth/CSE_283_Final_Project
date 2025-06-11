import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import shap
import logging
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recurrence_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RecurrenceClassifier:
    """A comprehensive classifier for breast cancer recurrence prediction"""
    
    def __init__(self, random_state=42, n_features=100):
        self.random_state = random_state
        self.n_features = n_features
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=random_state, class_weight='balanced'
            ),
            'xgboost': XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=random_state, max_iter=2000, class_weight='balanced', solver='liblinear'
            )
        }
        
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=n_features)
        self.is_fitted = False
        self.feature_names = None
        self.cv_results = {}
        
    def fit(self, X, y, feature_names=None):
        """Fit the classifier with cross-validation model selection"""
        logger.info("Starting model training...")
        logger.info(f"Initial data shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Adjust feature selection based on data size
        actual_k = min(self.n_features, X.shape[1], X.shape[0] // 2)
        self.feature_selector.set_params(k=actual_k)
        
        # Feature selection
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = self.feature_selector.get_support()
        selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_features[i]]
        
        logger.info(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
        logger.info(f"Top 10 selected features: {selected_feature_names[:10]}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Apply SMOTE if needed
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if n_positive >= 2 and n_negative >= 2:
            try:
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(3, min(n_positive, n_negative) - 1))
                X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
                logger.info(f"After SMOTE: {X_resampled.shape[0]} samples, class distribution: {np.bincount(y_resampled)}")
            except:
                logger.warning("SMOTE failed, using original data")
                X_resampled, y_resampled = X_scaled, y
        else:
            X_resampled, y_resampled = X_scaled, y
        
        # Cross-validation
        n_splits = min(5, len(y) // 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        best_score = 0
        best_model_name = None
        
        # Evaluate each model
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            scores = []
            
            try:
                for train_idx, val_idx in cv.split(X_resampled, y_resampled):
                    X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
                    y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                    scores.append(f1)
                
                avg_score = np.mean(scores)
                self.cv_results[name] = {'mean_f1': avg_score, 'std_f1': np.std(scores)}
                
                logger.info(f"{name}: CV F1-Score = {avg_score:.4f} (+/- {np.std(scores)*2:.4f})")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = name
                    
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
                continue
        
        if best_model_name is None:
            logger.error("No model could be successfully trained!")
            return self
        
        # Train best model on full dataset
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        self.best_model.fit(X_resampled, y_resampled)
        logger.info(f"Best model: {best_model_name} with CV F1-Score: {best_score:.4f}")
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        return self.best_model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        return self.best_model.predict_proba(X_scaled)

    def get_feature_importance(self, top_n=20):
        """Get feature importance from the best model"""
        if not self.is_fitted:
            return None
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                importances = np.abs(self.best_model.coef_[0])
            else:
                return None
            
            selected_features = self.feature_selector.get_support()
            selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_features[i]]
            
            feature_importance_df = pd.DataFrame({
                'feature': selected_feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return feature_importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None

def load_data_files(data_dir):
    """Load and validate the TPM and patient info files"""
    logger.info(f"Loading data from: {data_dir}")
    
    try:
        # Load TPM data
        tmp_file = data_dir / "pnas_tpm_96_nodup.txt"
        logger.info(f"Loading TPM file: {tmp_file}")
        
        if not tmp_file.exists():
            logger.error(f"TPM file not found: {tmp_file}")
            return None, None
        
        # Read TPM file with tab separator
        logger.info("Reading TPM file...")
        tmp_data = pd.read_csv(tmp_file, sep='\t', index_col=0)
        logger.info(f"TPM data shape: {tmp_data.shape}")
        
        # Transpose so samples are rows and genes are columns
        tmp_data = tmp_data.T
        logger.info(f"After transpose - TPM shape: {tmp_data.shape} (samples x genes)")
        
        # Load patient info
        patient_file = data_dir / "pnas_patient_info.csv"
        logger.info(f"Loading patient file: {patient_file}")
        
        if not patient_file.exists():
            logger.error(f"Patient info file not found: {patient_file}")
            return None, None
            
        patient_info = pd.read_csv(patient_file)
        logger.info(f"Patient info shape: {patient_info.shape}")
        logger.info(f"Patient info columns: {patient_info.columns.tolist()}")
        
        # Check for recurrence column
        if 'recurrence' not in patient_info.columns:
            logger.error("'recurrence' column not found in patient info file")
            logger.info(f"Available columns: {list(patient_info.columns)}")
            return None, None
        
        # Check dimensions
        if len(tmp_data) != len(patient_info):
            logger.warning(f"Dimension mismatch: TPM has {len(tmp_data)} samples, patient info has {len(patient_info)} samples")
            # Try to align by taking minimum
            min_samples = min(len(tmp_data), len(patient_info))
            tmp_data = tmp_data.iloc[:min_samples]
            patient_info = patient_info.iloc[:min_samples]
            logger.info(f"Aligned to {min_samples} samples")
        
        # Basic statistics
        logger.info(f"TPM data statistics:")
        logger.info(f"  Min value: {tmp_data.min().min():.4f}")
        logger.info(f"  Max value: {tmp_data.max().max():.4f}")
        logger.info(f"  Mean value: {tmp_data.mean().mean():.4f}")
        
        # Check target distribution
        recurrence_dist = patient_info['recurrence'].value_counts()
        logger.info(f"Recurrence distribution: {recurrence_dist.to_dict()}")
        
        return tmp_data, patient_info
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def prepare_features_and_target(tmp_data, patient_info):
    """Prepare features and target variables for modeling"""
    try:
        # Gene expression features
        X_gene = tmp_data.values
        gene_names = tmp_data.columns.tolist()
        
        logger.info(f"Gene expression features: {X_gene.shape}")
        
        # Clinical features if available
        clinical_features = []
        clinical_names = []
        
        for col in patient_info.columns:
            if col not in ['sample_id', 'patient_id', 'recurrence']:
                try:
                    if patient_info[col].dtype == 'object':
                        # Encode categorical variables
                        le = LabelEncoder()
                        encoded_values = le.fit_transform(patient_info[col].astype(str))
                        clinical_features.append(encoded_values)
                        clinical_names.append(f"{col}_encoded")
                        logger.info(f"Encoded categorical feature: {col}")
                    else:
                        # Use numerical features as-is
                        values = patient_info[col].fillna(patient_info[col].median())
                        clinical_features.append(values)
                        clinical_names.append(col)
                        logger.info(f"Added numerical feature: {col}")
                except Exception as e:
                    logger.warning(f"Could not process clinical feature {col}: {str(e)}")
        
        # Combine features
        if clinical_features:
            X_clinical = np.column_stack(clinical_features)
            X_combined = np.column_stack([X_gene, X_clinical])
            feature_names = gene_names + clinical_names
            logger.info(f"Combined features: {X_combined.shape} (genes: {len(gene_names)}, clinical: {len(clinical_names)})")
        else:
            X_combined = X_gene
            feature_names = gene_names
            logger.info("Using only gene expression features")
        
        # Target variable
        y = patient_info['recurrence'].values
        
        # Validate target
        unique_targets = np.unique(y)
        logger.info(f"Target variable - unique values: {unique_targets}")
        logger.info(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        if len(unique_targets) < 2:
            logger.error("Target variable must have at least 2 classes!")
            return None, None, None
        
        return X_combined, y, feature_names
        
    except Exception as e:
        logger.error(f"Error preparing features and target: {str(e)}")
        return None, None, None

def main():
    """Main function to run the breast cancer recurrence classification analysis"""
    logger.info("Starting Breast Cancer Recurrence Classification Analysis")
    logger.info("="*60)
    
    # Get data directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else Path.cwd()
    data_dir = script_dir / "data"
    
    logger.info(f"Data directory: {data_dir}")
    
    # Load data
    logger.info("Loading data...")
    tmp_data, patient_info = load_data_files(data_dir)
    
    if tmp_data is None or patient_info is None:
        logger.error("Failed to load data.")
        return None
    
    # Prepare features and target
    logger.info("Preparing features and target variables...")
    X, y, feature_names = prepare_features_and_target(tmp_data, patient_info)
    
    if X is None or y is None:
        logger.error("Failed to prepare features and target.")
        return None
    
    try:
        logger.info(f"Final dataset prepared:")
        logger.info(f"  Features shape: {X.shape}")
        logger.info(f"  Target shape: {y.shape}")
        logger.info(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Check if we have sufficient data
        if len(y) < 10:
            logger.warning("Very small dataset! Results may not be reliable.")
        
        if np.sum(y) == 0 or np.sum(y) == len(y):
            logger.error("All samples have the same label! Cannot train classifier.")
            return None
        
        # Initialize and train classifier
        logger.info("\nInitializing classifier...")
        classifier = RecurrenceClassifier(
            random_state=42, 
            n_features=min(100, X.shape[1] // 2, X.shape[0] - 1)
        )
        
        logger.info("Training classifier with cross-validation...")
        classifier.fit(X, y, feature_names)
        
        if not classifier.is_fitted:
            logger.error("Model training failed!")
            return None
        
        # Generate predictions and explanations
        logger.info("\nGenerating predictions and analysis...")
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Make predictions on test set
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        
        # Calculate performance metrics
        logger.info("\nFINAL MODEL PERFORMANCE:")
        logger.info("="*30)
        logger.info(f"Best model: {classifier.best_model_name}")
        logger.info(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"Test F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        logger.info(f"Test Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        logger.info(f"Test Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        
        # Try to calculate AUC if possible
        try:
            if len(np.unique(y_test)) > 1 and y_pred_proba.shape[1] > 1:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                logger.info(f"Test AUC: {auc_score:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {str(e)}")
        
        # Generate classification report
        logger.info("\nDetailed Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Get feature importance
        feature_importance = classifier.get_feature_importance(top_n=20)
        if feature_importance is not None:
            logger.info("\nTop 20 Most Important Features:")
            logger.info("="*40)
            for idx, row in feature_importance.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.6f}")
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Best performing model: {classifier.best_model_name}")
        logger.info(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
        
        return classifier, X_test, y_test, y_pred, y_pred_proba
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    result = main()
    if result is not None:
        logger.info("Script execution completed successfully!")
    else:
        logger.error("Script execution failed.")