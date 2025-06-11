import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def main():
    print("🎨 Creating Breast Cancer Classification Visualizations")
    print("=" * 60)
    
    try:
        # Load TPM data
        print("Loading TPM data...")
        file_path = Path("data/pnas_tpm_96_nodup.txt")
        data = pd.read_csv(file_path, sep='\t', index_col=0)
        data = data.T  # Transpose so samples are rows
        print(f"✓ TPM data loaded: {data.shape}")
        
        # Load patient info
        print("Loading patient info...")
        patient_file = Path("data/pnas_patient_info.csv")
        patients = pd.read_csv(patient_file)
        print(f"✓ Patient data loaded: {patients.shape}")
        
        # 1. Data Overview Plot
        print("\n1. Creating data overview plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Breast Cancer Dataset Overview', fontsize=16, fontweight='bold')
        
        # Recurrence distribution
        recur_counts = patients['recurrence'].value_counts()
        axes[0, 0].pie(recur_counts.values, labels=['No Recurrence', 'Recurrence'], 
                       autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Recurrence Distribution')
        
        # Age distribution
        axes[0, 1].hist(patients['age'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Age Distribution')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Frequency')
        
        # Stage distribution
        stage_counts = patients['stage'].value_counts()
        axes[1, 0].bar(stage_counts.index, stage_counts.values, color=['lightgreen', 'orange', 'lightcoral'])
        axes[1, 0].set_title('Cancer Stage Distribution')
        axes[1, 0].set_xlabel('Stage')
        axes[1, 0].set_ylabel('Count')
        
        # Gene expression summary
        axes[1, 1].hist(data.mean(axis=1), bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Average Gene Expression per Sample')
        axes[1, 1].set_xlabel('Mean Expression')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
        print("✓ Data overview saved as 'data_overview.png'")
        plt.show()
        
        # 2. Clinical Features Analysis
        print("\n2. Creating clinical analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Clinical Features vs Recurrence', fontsize=16, fontweight='bold')
        
        # Age by recurrence
        no_recur = patients[patients['recurrence'] == 0]['age']
        yes_recur = patients[patients['recurrence'] == 1]['age']
        axes[0, 0].hist([no_recur, yes_recur], bins=12, alpha=0.7, 
                        label=['No Recurrence', 'Recurrence'], color=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Age by Recurrence Status')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # Stage vs recurrence
        stage_recur = pd.crosstab(patients['stage'], patients['recurrence'])
        stage_recur_pct = stage_recur.div(stage_recur.sum(axis=1), axis=0) * 100
        stage_recur_pct.plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'lightcoral'])
        axes[0, 1].set_title('Recurrence Rate by Stage')
        axes[0, 1].set_xlabel('Stage')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].legend(['No Recurrence', 'Recurrence'])
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # Tumor size distribution
        patients.boxplot(column='tumor_size', by='recurrence', ax=axes[1, 0])
        axes[1, 0].set_title('Tumor Size by Recurrence')
        axes[1, 0].set_xlabel('Recurrence (0=No, 1=Yes)')
        axes[1, 0].set_ylabel('Tumor Size (cm)')
        
        # Biomarker analysis
        er_recur = pd.crosstab(patients['er_status'], patients['recurrence'])
        er_recur_pct = er_recur.div(er_recur.sum(axis=1), axis=0) * 100
        er_recur_pct.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Recurrence Rate by ER Status')
        axes[1, 1].set_xlabel('ER Status (0=Negative, 1=Positive)')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].legend(['No Recurrence', 'Recurrence'])
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('clinical_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Clinical analysis saved as 'clinical_analysis.png'")
        plt.show()
        
        # 3. Model Training and Performance
        print("\n3. Training models and creating performance plots...")
        
        # Prepare data
        X = data.values
        y = patients['recurrence'].values
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            results[name] = {'pred': y_pred, 'proba': y_pred_proba, 'model': model}
        
        # Performance plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # ROC Curves
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['proba'])
            roc_auc = auc(fpr, tpr)
            axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confusion Matrices
        for i, (name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['pred'])
            ax = axes[0, 1] if i == 0 else axes[1, 0]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Performance metrics comparison
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, result['pred']),
                'Precision': precision_score(y_test, result['pred']),
                'Recall': recall_score(y_test, result['pred']),
                'F1': f1_score(y_test, result['pred'])
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Plot metrics
        x = np.arange(len(metrics_df))
        width = 0.2
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, metric in enumerate(metrics_names):
            axes[1, 1].bar(x + i*width, metrics_df[metric], width, 
                          label=metric, color=colors[i], alpha=0.8)
        
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xticks(x + width * 1.5)
        axes[1, 1].set_xticklabels(metrics_df['Model'])
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("✓ Model performance saved as 'model_performance.png'")
        plt.show()
        
        # 4. Feature Importance
        print("\n4. Creating feature importance plot...")
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        
        # Get top 20 features
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
        plt.bar(range(20), importances[indices], color='steelblue', alpha=0.7)
        plt.xlabel('Feature Rank')
        plt.ylabel('Importance Score')
        plt.xticks(range(20), [f'Gene_{i+1}' for i in range(20)])
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance saved as 'feature_importance.png'")
        plt.show()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 ALL VISUALIZATIONS COMPLETED!")
        print("=" * 60)
        print("\nGenerated files:")
        print("• data_overview.png")
        print("• clinical_analysis.png") 
        print("• model_performance.png")
        print("• feature_importance.png")
        
        print(f"\n📊 Performance Summary:")
        print(metrics_df.round(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()