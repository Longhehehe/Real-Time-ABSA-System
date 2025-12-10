import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

from Src.train_multinb_rf import MultinomialNBModel
from Src.train_LLM import LLMSentimentClassifier

def train_full_ml_model(data, preprocessed_texts, vncorenlp_path):
    nb_full = MultinomialNBModel(vncorenlp_path, use_parallel=True)
    results_full, X_test_full, y_test_full, y_pred_full = nb_full.train_model_with_preprocessed(
        data, preprocessed_texts, use_count=True, use_feature_selection=True, use_svd=True
    )
    
    return {
        'results': results_full,
        'y_test': y_test_full,
        'y_pred': y_pred_full,
        'type': 'ML + Full Pipeline (Count + Chi2 FS + SVD)'
    }

def calculate_detailed_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrices(models_data, figsize=(14, 6)):
    n_models = len(models_data)
    if n_models == 0:
        return
    if n_models == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
    
    for i, (model_name, data) in enumerate(models_data.items()):
        if i >= len(axes):
            break
            
        y_true = data['y_test']
        y_pred = data['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Get unique labels and create proper label names
        labels = np.unique(np.concatenate([y_true, y_pred]))
        label_names = [f"Class {label}" for label in labels]
        
        # Create heatmap with better styling
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues', 
                   xticklabels=label_names, 
                   yticklabels=label_names, 
                   ax=axes[i],
                   cbar_kws={'shrink': 0.8},
                   square=True,
                   linewidths=0.5,
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        # Improve title and labels
        clean_name = model_name.replace('_', ' ')
        axes[i].set_title(f'{clean_name}\nConfusion Matrix', 
                         fontsize=14, fontweight='bold', pad=20)
        axes[i].set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Actual Labels', fontsize=12, fontweight='bold')
        
        # Improve tick labels
        axes[i].tick_params(axis='x', rotation=0, labelsize=11)
        axes[i].tick_params(axis='y', rotation=0, labelsize=11)
    
    # Overall title and layout
    plt.suptitle('Confusion Matrices Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def plot_metrics_comparison(all_metrics, figsize=(16, 10)):
    if not all_metrics:
        return
    
    df_metrics = pd.DataFrame(all_metrics)
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Set up the plot with professional styling
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    fig.patch.set_facecolor('white')
    
    # Professional color scheme
    model_colors = ['#2E86AB', '#F18F01']  # Blue for ML, Orange for PhoBERT
    metric_accent_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[i]
        
        # Create bars with gradient effect
        values = df_metrics[metric].tolist()
        bars = ax.bar(range(len(df_metrics)), values, 
                     color=[model_colors[j % len(model_colors)] for j in range(len(df_metrics))], 
                     alpha=0.85, edgecolor='white', linewidth=2)
        
        # Add subtle gradient and styling to bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            bar.set_facecolor(model_colors[j % len(model_colors)])
            bar.set_edgecolor('#34495E')
            bar.set_linewidth(1.5)
            
            # Add value labels with professional styling
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                   f'{value:.4f}', 
                   ha='center', va='bottom', 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='white', alpha=0.95,
                           edgecolor=model_colors[j % len(model_colors)], 
                           linewidth=1.5))
        
        # Customize subplot appearance
        ax.set_title(f'{name}', fontsize=15, fontweight='bold', 
                    pad=20, color='#2C3E50')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold', color='#34495E')
        ax.set_ylim(0, 1.08)  # Increased padding for labels
        
        # Add professional grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='y', color='#BDC3C7')
        ax.set_axisbelow(True)
        ax.set_facecolor('#FAFAFA')
        
        # Clean up model names and set labels
        model_labels = []
        for label in df_metrics['model']:
            if 'Full_ML_Pipeline' in label or 'ML' in label:
                model_labels.append('ML Pipeline')
            elif 'PhoBERT' in label or 'LLM' in label:
                model_labels.append('PhoBERT')
            else:
                model_labels.append(label.replace('_', ' '))
        
        ax.set_xticks(range(len(df_metrics)))
        ax.set_xticklabels(model_labels, fontsize=12, fontweight='bold', color='#2C3E50')
        ax.tick_params(axis='y', labelsize=10, colors='#7F8C8D')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
    
    # Professional title and subtitle
    plt.suptitle('Vietnamese Sentiment Analysis - Model Performance Comparison', 
                fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')
    
    # Add subtitle with model details
    model_info = []
    for _, row in df_metrics.iterrows():
        model_type = row.get('type', 'Unknown')
        model_info.append(f"{row['model'].replace('_', ' ')} ({model_type})")
    subtitle = ' vs '.join(model_info)
    fig.text(0.5, 0.94, subtitle, ha='center', fontsize=11, 
             style='italic', color='#7F8C8D')
    
    # Add professional legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[i], alpha=0.85, 
                           edgecolor='#34495E', linewidth=1,
                           label=model_labels[i]) for i in range(len(model_labels))]
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=len(model_labels), 
              fontsize=12, frameon=True, fancybox=True, 
              shadow=True, facecolor='white', edgecolor='#BDC3C7')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def load_and_preprocess_data(data_path, vncorenlp_path):
    try:
        data = pd.read_csv(data_path, quotechar='"', skipinitialspace=True)
    except pd.errors.ParserError:
        data = pd.read_csv(data_path, quotechar='"', skipinitialspace=True, on_bad_lines='skip')
    
    data.dropna(subset=['content', 'label'], inplace=True)
    
    # Preprocess texts once for all models
    from Src.preprocessing import preprocess_text
    preprocessed_texts = preprocess_text(data['content'].tolist(), use_parallel=True, n_workers=4)
    
    return data, preprocessed_texts

def main():
    data_path = "data.csv"
    vncorenlp_path = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"
    
    data = pd.read_csv(data_path, quotechar='"', skipinitialspace=True, on_bad_lines='skip')

    data.dropna(subset=['content', 'label'], inplace=True)

    # Load and preprocess data
    data, preprocessed_texts = load_and_preprocess_data(data_path, vncorenlp_path)
    
    # Initialize results dictionary
    models_results = {}
    
    full_ml_data = train_full_ml_model(data, preprocessed_texts, vncorenlp_path)
    models_results['Full_ML_Pipeline'] = full_ml_data
    
    phobert_model = LLMSentimentClassifier("vinai/phobert-base", vncorenlp_path)
    # Pass preprocessed texts to avoid re-preprocessing
    phobert_results = phobert_model.train_model_with_preprocessed(
        data, preprocessed_texts, epochs=6, batch_size=32, max_length=256
    )
    
    models_results['PhoBERT'] = {
        'results': phobert_results,
        'y_test': None,  # Would need to modify LLM class to get this
        'y_pred': None,  # Would need to modify LLM class to get this
        'type': 'LLM (Custom PyTorch FC Head)'
    }
    
    # Display results summary
    
    # Calculate detailed metrics
    detailed_metrics = []
    models_with_predictions = {}
    all_models_metrics = []  # For plotting both ML and PhoBERT
    
    for model_name, data in models_results.items():
        if data['y_test'] is not None and data['y_pred'] is not None:
            # For models with predictions (ML Pipeline)
            metrics = calculate_detailed_metrics(data['y_test'], data['y_pred'], model_name)
            metrics['type'] = data['type']
            metrics['training_time'] = data['results']['training_time']
            detailed_metrics.append(metrics)
            all_models_metrics.append(metrics)  # Add to combined list
            models_with_predictions[model_name] = data
        else:
            # For models without predictions (PhoBERT)
            basic_metrics = {
                'model': model_name,
                'type': data['type'],
                'accuracy': data['results']['accuracy'],
                'precision': data['results'].get('precision', data['results']['accuracy']),  # Use accuracy as fallback
                'recall': data['results'].get('recall', data['results']['accuracy']),     # Use accuracy as fallback
                'f1': data['results']['f1'],
                'training_time': data['results']['training_time']
            }
            all_models_metrics.append(basic_metrics)  # Add to combined list
            print(f"\n{model_name} Results:")
            print(f"   Accuracy: {basic_metrics['accuracy']:.4f}")
            print(f"   F1-Score: {basic_metrics['f1']:.4f}")
            print(f"   Training Time: {basic_metrics['training_time']:.2f}s")

    if all_models_metrics:
        # Create enhanced visualizations
        if models_with_predictions:
            plot_confusion_matrices(models_with_predictions, figsize=(15, 7))
            print("Confusion matrices saved as 'confusion_matrices.png'")
        
        plot_metrics_comparison(all_models_metrics, figsize=(14, 10))
        print("Metrics comparison saved as 'metrics_comparison.png'")
        
        if models_with_predictions:
            create_side_by_side_comparison(all_models_metrics, models_with_predictions, figsize=(18, 10))
            print("Comprehensive analysis saved as 'comprehensive_analysis.png'")
        
        # Model comparison
        analyze_two_models_comparison(all_models_metrics)

def analyze_two_models_comparison(all_metrics):
    df_metrics = pd.DataFrame(all_metrics)
    
    if len(df_metrics) >= 2:
        model1 = df_metrics.iloc[0]
        model2 = df_metrics.iloc[1]
        
        print(f"\nModel 1: {model1['model']}")
        print(f"   Accuracy:     {model1['accuracy']:.4f}")
        print(f"   Precision:    {model1['precision']:.4f}")
        print(f"   Recall:       {model1['recall']:.4f}")
        print(f"   F1-Score:     {model1['f1']:.4f}")
        print(f"   Training Time: {model1['training_time']:.2f}s")
        
        print(f"\nModel 2: {model2['model']}")
        print(f"   Accuracy:     {model2['accuracy']:.4f}")
        print(f"   Precision:    {model2['precision']:.4f}")
        print(f"   Recall:       {model2['recall']:.4f}")
        print(f"   F1-Score:     {model2['f1']:.4f}")
        print(f"   Training Time: {model2['training_time']:.2f}s")

def create_side_by_side_comparison(all_metrics, models_with_predictions, figsize=(20, 12)):

    # Create figure with professional layout
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Create custom grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[0.3, 1, 1], width_ratios=[1, 1, 1.2, 1.2], 
                         hspace=0.4, wspace=0.3)
    
    # Professional color scheme
    model_colors = ['#2E86AB', '#F18F01', '#A23B72', '#C73E1D']
    
    # Add title section
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(0.5, 0.7, 'Comprehensive Vietnamese Sentiment Analysis Comparison', 
                 ha='center', va='center', fontsize=20, fontweight='bold', 
                 color='#2C3E50', transform=title_ax.transAxes)
    
    df_metrics = pd.DataFrame(all_metrics)
    model_info = []
    for _, row in df_metrics.iterrows():
        model_type = row.get('type', 'Unknown')
        model_name = row['model'].replace('_', ' ')
        model_info.append(f"{model_name} ({model_type})")
    subtitle = ' vs '.join(model_info)
    
    title_ax.text(0.5, 0.3, subtitle, ha='center', va='center', 
                 fontsize=14, style='italic', color='#7F8C8D', 
                 transform=title_ax.transAxes)
    
    # Plot metrics comparison (top section)
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = fig.add_subplot(gs[1, i])
        
        values = df_metrics[metric].tolist()
        bars = ax.bar(range(len(df_metrics)), values, 
                     color=model_colors[:len(df_metrics)], alpha=0.85, 
                     edgecolor='white', linewidth=2)
        
        # Style individual bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            bar.set_edgecolor('#34495E')
            bar.set_linewidth(1.5)
            
            # Add value labels with professional styling
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.4f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.95,
                           edgecolor=model_colors[j % len(model_colors)], 
                           linewidth=1.5))
        
        # Customize subplot
        ax.set_title(f'{name}', fontsize=13, fontweight='bold', 
                    pad=15, color='#2C3E50')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold', color='#34495E')
        ax.set_ylim(0, 1.1)
        
        # Professional grid and styling
        ax.grid(True, alpha=0.3, linestyle='--', axis='y', color='#BDC3C7')
        ax.set_axisbelow(True)
        ax.set_facecolor('#FAFAFA')
        
        # Clean model names
        clean_names = []
        for name in df_metrics['model']:
            if 'Full_ML_Pipeline' in name or 'ML' in name:
                clean_names.append('ML Pipeline')
            elif 'PhoBERT' in name or 'LLM' in name:
                clean_names.append('PhoBERT')
            else:
                clean_names.append(name.replace('_', ' '))
        
        ax.set_xticks(range(len(df_metrics)))
        ax.set_xticklabels(clean_names, fontsize=10, fontweight='bold', color='#2C3E50')
        ax.tick_params(axis='y', labelsize=9, colors='#7F8C8D')
        
        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
    
    # Add performance summary table (bottom left)
    summary_ax = fig.add_subplot(gs[2, :2])
    summary_ax.axis('off')
    
    # Create summary statistics
    summary_data = []
    for _, row in df_metrics.iterrows():
        model_name = row['model'].replace('_', ' ')
        if 'Full_ML_Pipeline' in row['model'] or 'ML' in row['model']:
            model_name = 'ML Pipeline'
        elif 'PhoBERT' in row['model'] or 'LLM' in row['model']:
            model_name = 'PhoBERT'
        
        summary_data.append([
            model_name,
            f"{row['accuracy']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['training_time']:.2f}s"
        ])
    
    # Create table
    table = summary_ax.table(cellText=summary_data,
                           colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time'],
                           cellLoc='center',
                           loc='center',
                           colColours=['#E8F4FD']*6)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='#2C3E50')
            cell.set_facecolor('#3498DB')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#F8F9FA' if i % 2 == 0 else '#FFFFFF')
            cell.set_text_props(color='#2C3E50')
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(1)
    
    summary_ax.set_title('Performance Summary Table', fontsize=14, 
                        fontweight='bold', pad=20, color='#2C3E50')
    
    # Plot confusion matrix (bottom right)
    if models_with_predictions:
        cm_ax = fig.add_subplot(gs[2, 2:])
        
        # Get first model with predictions (usually ML Pipeline)
        first_model = list(models_with_predictions.items())[0]
        model_name, data = first_model
        
        y_true = data['y_test']
        y_pred = data['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
        # Create class name mapping
        class_names = []
        for label in labels:
            if label == 0:
                class_names.append('Negative')
            elif label == 1:
                class_names.append('Positive')
            else:
                class_names.append(f'Class {label}')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu_r', 
                   xticklabels=class_names, yticklabels=class_names, 
                   ax=cm_ax, square=True, linewidths=2,
                   cbar_kws={'shrink': 0.8, 'aspect': 20, 'pad': 0.02},
                   annot_kws={'size': 14, 'weight': 'bold', 'color': '#2C3E50'})
        
        clean_name = model_name.replace('_', ' ')
        if 'Full_ML_Pipeline' in model_name or 'ML' in model_name:
            clean_name = 'ML Pipeline'
        
        cm_ax.set_title(f'{clean_name} - Confusion Matrix', 
                       fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
        cm_ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold', color='#34495E')
        cm_ax.set_ylabel('Actual Labels', fontsize=12, fontweight='bold', color='#34495E')
        cm_ax.tick_params(axis='both', labelsize=11, colors='#2C3E50')
    
    # Add professional legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[i], alpha=0.85, 
                           edgecolor='#34495E', linewidth=1,
                           label=clean_names[i]) for i in range(len(clean_names))]
    
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=len(clean_names), 
              fontsize=12, frameon=True, fancybox=True, 
              shadow=True, facecolor='white', edgecolor='#BDC3C7')
    
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
if __name__ == "__main__":
    main()
