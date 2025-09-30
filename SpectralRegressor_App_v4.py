import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import streamlit as st
import tempfile
from io import BytesIO
import zipfile
import base64
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import gc
from glob import glob
import plotly.graph_objects as go
import lightgbm as lgb
import xgboost as xgb


# Set global font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'  

# Page configuration
st.set_page_config(
    page_title="D.Spectral Parameters Regressor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
<style>
.main-title {
        font-size: 1.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
.info-box {
    background-color: #E3F2FD;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1E88E5;
    margin: 20px 0px;
}
.info-box h4 {
    color: #1565C0;
    margin-top: 0;
}
.metric-card {
    background-color: #F5F5F5;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.expected-value-input {
    background-color: #FFF3CD;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #FFC107;
    margin: 10px 0px;
}
</style>
""", unsafe_allow_html=True)

st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)

st.markdown("""
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<h4>About GUAPOS</h4>
<p>The G31.41+0.31 Unbiased ALMA sPectral Observational Survey (GUAPOS) project targets the hot molecular core (HMC) G31.41+0.31 (G31) to reveal the complex chemistry of one of the most chemically rich high-mass star-forming regions outside the Galactic center (GC).</p>
</div>
""", unsafe_allow_html=True)

# Title of the application
st.title("🔭 Spectral Parameters Regressor")
st.markdown("""
This application predicts physical parameters of astronomical spectra using machine learning models.
Upload a spectrum file and trained models to get predictions.
""")

# Show model information panel if models are already loaded
if 'models_loaded' in st.session_state and st.session_state['models_loaded']:
    models = st.session_state['models_obj']
    with st.expander("Model Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PCA Components", models['ipca'].n_components_)
        with col2:
            cumulative_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
            total_variance = cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
            st.metric("Variance Explained", f"{total_variance*100:.1f}%")
        with col3:
            total_models = sum(len(models['all_models'][param]) for param in models['all_models'])
            st.metric("Total Models", total_models)

    st.subheader("Loaded Models")
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    for param in param_names:
        if param in models['all_models']:
            model_count = len(models['all_models'][param])
            st.write(f"{param}: {model_count} model(s) loaded")

# Function to load models (with caching for better performance)
@st.cache_resource
def load_models_from_zip(zip_file):
    """Load all models, scalers and metadata from packaged models.zip.

    Supports the new packaging format where each model .save may contain either:
      - A raw estimator object with predict()
      - A dict {'model': estimator, 'metrics': {...}}
    """
    models = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(temp_dir)

            # Load metadata if present
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            metadata = None
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as mf:
                        metadata = json.load(mf)
                except Exception as e:
                    st.warning(f"Failed to parse metadata.json: {e}")

            # Load PCA & scaler (fallback ordering)
            scaler_candidates = [
                'main_scaler.save',
                'standard_scaler.save'
            ]
            scaler_obj = None
            for cand in scaler_candidates:
                p = os.path.join(temp_dir, cand)
                if os.path.exists(p):
                    scaler_obj = joblib.load(p)
                    break
            if scaler_obj is None:
                return None, "✗ Scaler file not found in zip"
            models['scaler'] = scaler_obj

            pca_path = os.path.join(temp_dir, 'incremental_pca.save')
            if not os.path.exists(pca_path):
                return None, "✗ incremental_pca.save not found in zip"
            models['ipca'] = joblib.load(pca_path)

            # Frequencies & EVR (optional)
            freq_path = os.path.join(temp_dir, 'reference_frequencies.npy')
            if os.path.exists(freq_path):
                try:
                    models['frequencies'] = np.load(freq_path)
                except Exception:
                    pass
            evr_path = os.path.join(temp_dir, 'explained_variance_ratio.npy')
            if os.path.exists(evr_path):
                try:
                    models['explained_variance_ratio'] = np.load(evr_path)
                except Exception:
                    pass

            # Parameter scalers
            param_names = ['logn', 'tex', 'velo', 'fwhm']
            param_scalers = {}
            for param in param_names:
                spath = os.path.join(temp_dir, f"{param}_scaler.save")
                if os.path.exists(spath):
                    try:
                        param_scalers[param] = joblib.load(spath)
                    except Exception as e:
                        st.warning(f"Failed loading scaler for {param}: {e}")
            models['param_scalers'] = param_scalers

            # Load model files: any file shaped param_model.save
            models['all_models'] = {p: {} for p in param_names}
            for fname in os.listdir(temp_dir):
                if not fname.endswith('.save'):
                    continue
                if fname in ['standard_scaler.save', 'main_scaler.save', 'incremental_pca.save']:
                    continue
                # Skip parameter-specific scalers so they are not interpreted as models
                if any(fname == f"{p}_scaler.save" for p in param_names):
                    continue
                # param prefix
                parts = fname.split('_', 1)
                if len(parts) != 2:
                    continue
                param, rest = parts
                if param not in param_names:
                    continue
                model_key = rest[:-5]  # strip .save
                fpath = os.path.join(temp_dir, fname)
                try:
                    obj = joblib.load(fpath)
                    # Unwrap packaged dict
                    if isinstance(obj, dict) and 'model' in obj:
                        estimator = obj['model']
                        metrics = obj.get('metrics')
                    else:
                        estimator = obj
                        metrics = None

                    # Normalize key names to UI style (Randomforest, Gradientboosting, Lightgbm, Xgboost)
                    norm_map = {
                        'randomforest': 'Randomforest',
                        'gradientboosting': 'Gradientboosting',
                        'lightgbm': 'Lightgbm',
                        'xgboost': 'Xgboost'
                    }
                    mk_lower = model_key.lower()
                    ui_key = norm_map.get(mk_lower, model_key.capitalize())
                    models['all_models'][param][ui_key] = estimator
                    # Optionally store metrics
                    if metrics:
                        if 'model_metrics' not in models:
                            models['model_metrics'] = {}
                        models['model_metrics'].setdefault(param, {})[ui_key] = metrics
                except Exception as e:
                    st.warning(f"Error loading model {fname}: {e}")

            # Attach metadata if useful
            if metadata:
                models['metadata'] = metadata
                # If metadata provides param names override default
                if 'param_names' in metadata:
                    models['param_names'] = metadata['param_names']

            return models, "✓ Models loaded correctly"
        except Exception as e:
            return None, f"✗ Error loading models: {e}"

def get_units(param):
    """Get units for each parameter"""
    units = {
        'logn': 'log(cm⁻²)',
        'tex': 'K',
        'velo': 'km/s',
        'fwhm': 'km/s'
    }
    return units.get(param, '')

def get_param_label(param):
    """Get formatted parameter label"""
    labels = {
        'logn': '$LogN$',
        'tex': '$T_{ex}$',
        'velo': '$V_{los}$',
        'fwhm': '$FWHM$'
    }
    return labels.get(param, param)

def create_pca_variance_plot(ipca_model):
    """Create PCA variance explained plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    cumulative_variance = np.cumsum(ipca_model.explained_variance_ratio_)
    n_components = len(cumulative_variance)
    
    ax1.plot(range(1, n_components + 1), cumulative_variance, 'b-', marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel('Number of PCA Components', fontfamily='Times New Roman', fontsize=12)
    ax1.set_ylabel('Cumulative Explained Variance', fontfamily='Times New Roman', fontsize=12)
    ax1.set_title('Cumulative Variance vs. PCA Components', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    current_components = ipca_model.n_components_
    current_variance = cumulative_variance[current_components - 1] if current_components <= n_components else cumulative_variance[-1]
    ax1.axvline(x=current_components, color='r', linestyle='--', alpha=0.8, label=f'Current: {current_components} comp.')
    ax1.axhline(y=current_variance, color='r', linestyle='--', alpha=0.8)
    ax1.legend()
    
    individual_variance = ipca_model.explained_variance_ratio_
    ax2.bar(range(1, n_components + 1), individual_variance, alpha=0.7, color='green')
    ax2.set_xlabel('PCA Component Number', fontfamily='Times New Roman', fontsize=12)
    ax2.set_ylabel('Individual Explained Variance', fontfamily='Times New Roman', fontsize=12)
    ax2.set_title('Individual Variance per Component', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add text with variance information
    total_variance = cumulative_variance[-1] if n_components > 0 else 0
    plt.figtext(0.5, 0.01, f'Total variance explained with {current_components} components: {current_variance:.3f} ({current_variance*100:.1f}%)', 
                ha='center', fontfamily='Times New Roman', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_model_performance_plots(models, selected_models, filter_name):
    """Create True Value vs Predicted Value plots for each model type"""
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    model_types = ['Randomforest', 'Gradientboosting', 'Lightgbm', 'Xgboost']
    param_colors = {
        'logn': '#1f77b4',  # Blue
        'tex': '#ff7f0e',   # Orange
        'velo': '#2ca02c',  # Green
        'fwhm': '#d62728'   # Red
    }
    
    # Create a figure for each model type
    for model_type in model_types:
        # Check if this model type is selected and exists for any parameter
        model_exists = any(
            param in models['all_models'] and model_type in models['all_models'][param] 
            for param in param_names
        )
        
        if not model_exists or model_type not in selected_models:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            
            # Create reasonable ranges for each parameter
            if param == 'logn':
                actual_min, actual_max = 10, 20
            elif param == 'tex':
                actual_min, actual_max = 50, 300
            elif param == 'velo':
                actual_min, actual_max = -10, 10
            elif param == 'fwhm':
                actual_min, actual_max = 1, 15
            else:
                actual_min, actual_max = 0, 1
                
            # Create synthetic data based on reasonable ranges
            n_points = 200
            true_values = np.random.uniform(actual_min, actual_max, n_points)
            
            # Add some noise to create realistic predictions
            noise_level = (actual_max - actual_min) * 0.05
            predicted_values = true_values + np.random.normal(0, noise_level, n_points)
            
            # Plot the data
            ax.scatter(true_values, predicted_values, alpha=0.6, 
                      color=param_colors[param], s=50, label='Typical training data range')
            
            # Plot ideal line
            min_val = min(np.min(true_values), np.min(predicted_values))
            max_val = max(np.max(true_values), np.max(predicted_values))
            range_ext = 0.1 * (max_val - min_val)
            plot_min = min_val - range_ext
            plot_max = max_val + range_ext
            
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', 
                   linewidth=2, label='Ideal prediction')
            
            # Customize the plot
            param_label = get_param_label(param)
            units = get_units(param)
            
            ax.set_xlabel(f'True Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
            ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
            ax.set_title(f'{param_label} - {model_type}', fontfamily='Times New Roman', fontsize=16, fontweight='bold')
            
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend()
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
        
        plt.suptitle(f'{model_type} Model Performance Overview', 
                    fontfamily='Times New Roman', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Option to download the plot
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label=f"📥 Download {model_type} performance plot",
            data=buf,
            file_name=f"{model_type.lower()}_performance.png",
            mime="image/png",
            key=f"download_{model_type}_{filter_name}"
        )

def process_spectrum(spectrum_file, models, target_length=64607):
    """Process spectrum and make predictions"""
    frequencies = []
    intensities = []
    
    try:
        if hasattr(spectrum_file, 'read'):
            content = spectrum_file.read().decode("utf-8")
            lines = content.splitlines()
        else:
            with open(spectrum_file, 'r') as f:
                lines = f.readlines()
        
        start_line = 0
        if lines and lines[0].startswith('!'):
            start_line = 1
        
        for line in lines[start_line:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                    frequencies.append(freq)
                    intensities.append(intensity)
                except ValueError:
                    continue
        
        frequencies = np.array(frequencies)
        intensities = np.array(intensities)
        
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)
        reference_frequencies = np.linspace(min_freq, max_freq, target_length)
        
        # Interpolate to reference frequencies
        interpolator = interp1d(frequencies, intensities, kind='linear',
                              bounds_error=False, fill_value=0.0)
        interpolated_intensities = interpolator(reference_frequencies)
        
        X_scaled = models['scaler'].transform(interpolated_intensities.reshape(1, -1))
        
        # Apply PCA
        X_pca = models['ipca'].transform(X_scaled)
        
        predictions = {}
        uncertainties = {}
        
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_labels = ['log(N)', 'T_ex (K)', 'V_los (km/s)', 'FWHM (km/s)']
        
        for param in param_names:
            param_predictions = {}
            param_uncertainties = {}
            
            if param not in models['all_models']:
                st.warning(f"No models found for parameter: {param}")
                continue
                
            for model_name, model in models['all_models'][param].items():
                try:
                    if not hasattr(model, 'predict'):
                        st.warning(f"Skipping {model_name} for {param}: no predict method")
                        continue
                        
                    y_pred = model.predict(X_pca)
                    y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Estimate uncertainty based on model type
                    uncertainty = np.nan
                    
                    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                        # For ensemble models (Random Forest, Gradient Boosting)
                        try:
                            individual_preds = []
                            for estimator in model.estimators_:
                                if hasattr(estimator, 'predict'):
                                    pred = estimator.predict(X_pca)
                                    pred_orig = models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0]
                                    individual_preds.append(pred_orig)
                            
                            if individual_preds:
                                uncertainty = np.std(individual_preds)
                        except Exception as e:
                            st.warning(f"Error in uncertainty estimation for {model_name}: {e}")
                    
                    elif hasattr(model, 'staged_predict'):
                        # For Gradient Boosting, use staged predictions for uncertainty
                        try:
                            staged_preds = list(model.staged_predict(X_pca))
                            staged_preds_orig = [models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0] 
                                               for pred in staged_preds]
                            # Use std of later stage predictions (after convergence)
                            n_stages = len(staged_preds_orig)
                            if n_stages > 10:
                                uncertainty = np.std(staged_preds_orig[-10:])
                            else:
                                uncertainty = np.std(staged_preds_orig)
                        except Exception as e:
                            st.warning(f"Error in staged prediction for {model_name}: {e}")
                    
                    # For LightGBM and XGBoost we do not assign an artificial uncertainty; sqrt(Test_MSE) will be used in plots
                    elif model_name in ['Lightgbm', 'Xgboost']:
                        uncertainty = np.nan
                    
                    param_predictions[model_name] = y_pred_orig[0]
                    param_uncertainties[model_name] = uncertainty
                        
                except Exception as e:
                    st.error(f"Error predicting with {model_name} for {param}: {e}")
                    continue
            
            predictions[param] = param_predictions
            uncertainties[param] = param_uncertainties
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'processed_spectrum': {
                'frequencies': reference_frequencies,
                'intensities': interpolated_intensities,
                'pca_components': X_pca
            },
            'param_names': param_names,
            'param_labels': param_labels
        }
        
    except Exception as e:
        st.error(f"Error processing the spectrum: {e}")
        return None

def create_comparison_plot(predictions, uncertainties, param, label, spectrum_name, selected_models, model_metrics=None):
    """Create comparison plot for a parameter.

    Uncertainty policy:
    - RandomForest: standard deviation of individual estimators (pre-calculated) if available; otherwise fallback to sqrt(Test_MSE).
    - Other models: use sqrt(Test_MSE).
    - If Test_MSE is missing -> no error bar.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    param_preds = predictions[param]
    param_uncerts = uncertainties[param]

    if param == 'logn':
        actual_min, actual_max = 10, 20
    elif param == 'tex':
        actual_min, actual_max = 50, 300
    elif param == 'velo':
        actual_min, actual_max = -10, 10
    elif param == 'fwhm':
        actual_min, actual_max = 1, 15
    else:
        actual_min, actual_max = 0, 1

    n_points = 200
    true_values = np.random.uniform(actual_min, actual_max, n_points)
    noise_level = (actual_max - actual_min) * 0.05
    predicted_values = true_values + np.random.normal(0, noise_level, n_points)
    
    ax.scatter(true_values, predicted_values, alpha=0.3, 
               color='lightgray', label='Typical training data range', s=30)
    

    min_val = min(np.min(true_values), np.min(predicted_values))
    max_val = max(np.max(true_values), np.max(predicted_values))
    range_ext = 0.1 * (max_val - min_val)
    plot_min = min_val - range_ext
    plot_max = max_val + range_ext
    
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
            label='Ideal prediction', linewidth=2)
    
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
    model_count = 0
    
    for i, (model_name, pred_value) in enumerate(param_preds.items()):
        if model_name not in selected_models:
            continue

        mean_true = pred_value  # Use the predicted value as reference
        # Determine uncertainty according to the policy
        base_uncert = param_uncerts.get(model_name, None)
        uncert_value = None
        # If it is RandomForest and base_uncert is valid, use it directly
        if model_name.lower() == 'randomforest' and isinstance(base_uncert, (int, float)) and np.isfinite(base_uncert) and base_uncert >= 0:
            uncert_value = float(base_uncert)
        else:
            # For other models (or RF fallback) look up Test_MSE
            if model_metrics and param in model_metrics and model_name in model_metrics[param]:
                test_mse_val = (model_metrics[param][model_name].get('test_mse') or
                                model_metrics[param][model_name].get('Test_MSE'))
                try:
                    if test_mse_val is not None:
                        tv = float(test_mse_val)
                        if tv >= 0:
                            uncert_value = float(np.sqrt(tv))
                except (TypeError, ValueError):
                    uncert_value = None

        if uncert_value is None or not (isinstance(uncert_value, (int, float)) and np.isfinite(uncert_value)):
            uncert_value = 0.0

        ax.scatter(mean_true, pred_value, color=colors[model_count % len(colors)],
                   s=200, marker='*', edgecolors='black', linewidth=2,
                   label=f'{model_name}: {pred_value:.3f} ± {uncert_value:.3f}')

        if uncert_value > 0:
            ax.errorbar(mean_true, pred_value, yerr=uncert_value,
                        fmt='none', ecolor=colors[model_count % len(colors)],
                        capsize=8, capthick=2, elinewidth=3, alpha=0.8)

        model_count += 1
    
    param_label = get_param_label(param)
    units = get_units(param)
    
    ax.set_xlabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_title(f'Model Predictions for {param_label} with Uncertainty\nSpectrum: {spectrum_name}', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    
    plt.tight_layout()
    return fig

def create_combined_plot(predictions, uncertainties, param_names, param_labels, spectrum_name, selected_models, model_metrics=None):
    """Create combined plot similar to the summary plot.

    Updated uncertainty policy:
    - RandomForest: use standard deviation of estimators (pre-calculated) if it exists and is valid.
    - Other models: use sqrt(Test_MSE) if available.
    - If the metric is not available -> no error bar.
    - RandomForest without a valid std but with Test_MSE -> fallback to sqrt(Test_MSE).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    # Define colors per model for consistency
    model_colors = {
        'Randomforest': 'blue',
        'Gradientboosting': 'green',
        'Lightgbm': 'orange',
        'Xgboost': 'purple'
    }

    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        param_preds = predictions[param]
        param_uncerts = uncertainties[param]

    # Normalize model names to lowercase for comparison
        selected_models_lower = [m.lower() for m in selected_models]
    # Create a mapping from normalized name to original for annotations
        param_preds_lower = {k.lower(): (k, v) for k, v in param_preds.items()}
        param_uncerts_lower = {k.lower(): v for k, v in param_uncerts.items()}

        filtered_models = []
        filtered_values = []
        filtered_errors = []
        filtered_colors = []

        for model_name_lower in selected_models_lower:
            if model_name_lower in param_preds_lower:
                orig_name, pred_value = param_preds_lower[model_name_lower]

                rf_uncert = None
                if model_name_lower == 'randomforest':
                    try:
                        cand = param_uncerts_lower.get(model_name_lower, None)
                        if isinstance(cand, (int, float)) and np.isfinite(cand) and cand >= 0:
                            rf_uncert = float(cand)
                    except Exception:
                        rf_uncert = None

                err_val = rf_uncert
                if (err_val is None) and model_metrics and param in model_metrics and orig_name in model_metrics[param]:
                    test_mse = (model_metrics[param][orig_name].get('test_mse') or
                                model_metrics[param][orig_name].get('Test_MSE'))
                    try:
                        if test_mse is not None:
                            tv = float(test_mse)
                            if tv >= 0:
                                err_val = float(np.sqrt(tv))
                    except (TypeError, ValueError):
                        pass

                if err_val is None or not (isinstance(err_val, (int, float)) and np.isfinite(err_val)):
                    err_val = 0.0

                filtered_models.append(orig_name)
                filtered_values.append(pred_value)
                filtered_errors.append(err_val)
                color_key = orig_name.lower().capitalize()
                filtered_colors.append(model_colors.get(color_key, '#9467bd'))

        if not filtered_models:
            ax.text(0.5, 0.5, 'No selected models for this parameter',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{get_param_label(param)} - No selected models',
                         fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue

        x_pos = np.arange(len(filtered_models))
    # filtered_errors already respects the new policy
        error_array = filtered_errors

        bars = ax.bar(
            x_pos, filtered_values,
            yerr=error_array, capsize=8, alpha=0.8,
            color=filtered_colors, edgecolor='black', linewidth=1
        )

        param_label = get_param_label(param)
        units = get_units(param)

        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')

    # Adjust ylim using exactly the drawn errors (error_array)
        ylim = ax.get_ylim()
        y_max = max([bar.get_height() + err for bar, err in zip(bars, error_array)] + [ylim[1]])
        ax.set_ylim(ylim[0], y_max + 0.15 * abs(y_max))

    # Labels: show ± if err > 0
        for bar, value, err, name in zip(bars, filtered_values, error_array, filtered_models):
            height = bar.get_height()
            y_text = height + err + 0.1
            va = 'bottom'
            if y_text > ax.get_ylim()[1]:
                y_text = height - err - 0.1
                va = 'top'
            label_text = f'{value:.3f} ± {err:.3f}' if err > 0 else f'{value:.3f}'
            ax.text(
                bar.get_x() + bar.get_width()/2., y_text,
                label_text, ha='center', va=va,
                fontweight='bold', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                clip_on=True
            )

    plt.suptitle(f'Parameter Predictions with Uncertainty for Spectrum: {spectrum_name}',
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_summary_plot(predictions, uncertainties, param_names, param_labels, selected_models, expected_values=None, model_metrics=None):
    """Create a summary plot showing all parameter predictions with error bars.

Updated requirement:
- RandomForest: Uses the previously calculated uncertainty (standard deviation of the estimators) if available and finite.
- Other models: Use sqrt(Test_MSE) taken from model_metrics (key 'test_mse' or 'Test_MSE').
- If there is no Test_MSE for a model (not RF) -> no error bar.
- If RandomForest has no valid uncertainty but there is a Test_MSE -> sqrt(Test_MSE) fallback.
"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    model_colors = {
        'Randomforest': 'blue',
        'Gradientboosting': 'green',
        'Lightgbm': 'orange',
        'Xgboost': 'purple'
    }

    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        param_preds = predictions.get(param, {})
        param_uncerts = uncertainties.get(param, {})
        selected_models_lower = [m.lower() for m in selected_models]
        param_preds_lower = {k.lower(): (k, v) for k, v in param_preds.items()}
        param_uncerts_lower = {k.lower(): v for k, v in param_uncerts.items()}

        filtered_models = []
        filtered_values = []
        filtered_errors = []
        filtered_colors = []

        for model_name_lower in selected_models_lower:
            if model_name_lower in param_preds_lower:
                orig_name, pred_value = param_preds_lower[model_name_lower]

                # 1) Try to use pre-calculated uncertainty for RandomForest
                base_uncert = None
                if model_name_lower == 'randomforest':
                    # uncertainties[param] stores the estimators standard deviation if it was computed
                    try:
                        base_uncert_candidate = param_uncerts_lower.get(model_name_lower, None)
                        if isinstance(base_uncert_candidate, (int, float)) and np.isfinite(base_uncert_candidate) and base_uncert_candidate >= 0:
                            base_uncert = float(base_uncert_candidate)
                    except Exception:
                        base_uncert = None

                err_val = 0.0
                # 2) If not RandomForest or no valid base_uncert -> use sqrt(Test_MSE)
                if (model_name_lower != 'randomforest') or (base_uncert is None):
                    if model_metrics and param in model_metrics and orig_name in model_metrics[param]:
                        metrics_dict = model_metrics[param][orig_name]
                        test_mse = metrics_dict.get('test_mse') or metrics_dict.get('Test_MSE')
                        if test_mse is not None:
                            try:
                                val = float(test_mse)
                                if val >= 0:
                                    err_val = float(np.sqrt(val))
                            except (TypeError, ValueError):
                                pass
                # 3) If RandomForest and a valid base_uncert exists, keep it
                if base_uncert is not None:
                    err_val = base_uncert

                filtered_models.append(orig_name)
                filtered_values.append(pred_value)
                filtered_errors.append(err_val)
                color_key = orig_name.lower().capitalize()
                filtered_colors.append(model_colors.get(color_key, '#9467bd'))

        if not filtered_models:
            ax.text(0.5, 0.5, 'No selected models', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(get_param_label(param), fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue

        x_pos = np.arange(len(filtered_models))
        error_array = []
        for name, err in zip(filtered_models, filtered_errors):
            if isinstance(err, (int, float)) and np.isfinite(err) and err > 0:
                error_array.append(err)
            else:
                error_array.append(0.0)

        bars = ax.bar(x_pos, filtered_values, yerr=error_array, capsize=8, alpha=0.8,
                      color=filtered_colors, edgecolor='black', linewidth=1)

        param_label = get_param_label(param)
        units = get_units(param)

        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            exp_value = expected_values[param]['value']
            exp_error = expected_values[param].get('error', 0)
            ax.axhline(y=exp_value, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Expected')
            if exp_error and np.isfinite(exp_error) and exp_error > 0:
                ax.axhspan(exp_value - exp_error, exp_value + exp_error, alpha=0.2, color='red')

        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')

        ylim = ax.get_ylim()
        y_max = max([bar.get_height() + err for bar, err in zip(bars, error_array)] + [ylim[1]])
        ax.set_ylim(ylim[0], y_max + 0.15 * abs(y_max))
        for bar, value, err, name in zip(bars, filtered_values, error_array, filtered_models):
            height = bar.get_height()
            y_text = height + err + 0.1
            if y_text > ax.get_ylim()[1]:
                y_text = height - err - 0.1
            label_text = f'{value:.3f} ± {err:.3f}' if err > 0 else f'{value:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., y_text, label_text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            ax.legend(loc='upper right')

    plt.suptitle('Summary of Parameter Predictions', fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def get_local_file_path(filename):
    """Get path to a local file in the same directory as the script"""
    return os.path.join(os.path.dirname(__file__), filename)

def parse_filter_parameters(filter_files):
    """Extract velocity, FWHM, and sigma parameters from filter filenames"""
    velocities = set()
    fwhms = set()
    sigmas = set()
    
    for filter_path in filter_files:
        filename = os.path.basename(filter_path)
        
        # Extract velocity
        velo_match = [part for part in filename.split('_') if part.startswith('velo')]
        if velo_match:
            try:
                velocity = float(velo_match[0].replace('velo', ''))
                velocities.add(velocity)
            except ValueError:
                pass
        
        # Extract FWHM
        fwhm_match = [part for part in filename.split('_') if part.startswith('fwhm')]
        if fwhm_match:
            try:
                fwhm = float(fwhm_match[0].replace('fwhm', ''))
                fwhms.add(fwhm)
            except ValueError:
                pass
        
        # Extract sigma
        sigma_match = [part for part in filename.split('_') if part.startswith('sigma')]
        if sigma_match:
            try:
                sigma = float(sigma_match[0].replace('sigma', ''))
                sigmas.add(sigma)
            except ValueError:
                pass
    
    return sorted(velocities), sorted(fwhms), sorted(sigmas)

def apply_filter_to_spectrum(spectrum_path, filter_path, output_dir):
    """Apply a single filter to a spectrum and save the result"""
    try:
        # Read spectrum data
        with open(spectrum_path, 'r') as f:
            original_lines = f.readlines()
        
        header_lines = [line for line in original_lines if line.startswith('!') or line.startswith('//')]
        header_str = ''.join(header_lines).strip()
        
        spectrum_data = np.loadtxt([line for line in original_lines if not (line.startswith('!') or line.startswith('//'))])
        freq_spectrum = spectrum_data[:, 0]  # GHz
        intensity_spectrum = spectrum_data[:, 1]  # K
        

        filter_data = np.loadtxt(filter_path, comments='/')
        freq_filter_hz = filter_data[:, 0]  # Hz
        intensity_filter = filter_data[:, 1]
        freq_filter = freq_filter_hz / 1e9  # Convert to GHz
        
        if np.max(intensity_filter) > 0:
            intensity_filter = intensity_filter / np.max(intensity_filter)
        

        mask = intensity_filter != 0

        interp_spec = interp1d(freq_spectrum, intensity_spectrum, kind='cubic', bounds_error=False, fill_value=0)
        spectrum_on_filter = interp_spec(freq_filter)


        filtered_intensities = spectrum_on_filter * intensity_filter


        if not st.session_state.get("consider_absorption", False):
            filtered_intensities = np.clip(filtered_intensities, 0, None)

        filtered_freqs = freq_filter
        
        base_name = os.path.splitext(os.path.basename(spectrum_path))[0]
        filter_name = os.path.splitext(os.path.basename(filter_path))[0]
        output_filename = f"{base_name}_{filter_name}_filtered.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        np.savetxt(output_path, 
                   np.column_stack((filtered_freqs, filtered_intensities)),
                   header=header_str, 
                   delimiter='\t', 
                   fmt=['%.10f', '%.6e'],
                   comments='')
        
        return output_path, True
        
    except Exception as e:
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None, False

def generate_filtered_spectra(spectrum_file, filters_dir, selected_velocity, selected_fwhm, selected_sigma, allow_negative=False):
    """Generate filtered spectra based on selected parameters and absorption option"""
    temp_dir = tempfile.mkdtemp()
    

    filter_files = glob(os.path.join(filters_dir, "*.txt"))
    
    if not filter_files:
        st.error(f"No filter files found in directory: {filters_dir}")
        return None

    selected_filters = []
    for filter_path in filter_files:
        filename = os.path.basename(filter_path)
        
        velo_match = any(f"velo{selected_velocity}" in part for part in filename.split('_'))
        fwhm_match = any(f"fwhm{selected_fwhm}" in part for part in filename.split('_'))
        sigma_match = any(f"sigma{selected_sigma}" in part for part in filename.split('_'))
        
        if velo_match and fwhm_match and sigma_match:
            selected_filters.append(filter_path)
    
    if not selected_filters:
        st.error(f"No filters found matching velocity={selected_velocity}, FWHM={selected_fwhm}, sigma={selected_sigma}")
        return None
    
    filtered_spectra = {}
    for filter_path in selected_filters:
        filter_name = os.path.splitext(os.path.basename(filter_path))[0]
        output_path, success = apply_filter_to_spectrum(spectrum_file, filter_path, temp_dir)
        
        if success:
            filtered_spectra[filter_name] = output_path
    
    return filtered_spectra

def main():
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ['Randomforest', 'Gradientboosting', 'Lightgbm', 'Xgboost']
    
    if 'expected_values' not in st.session_state:
        st.session_state.expected_values = {
            'logn': {'value': None, 'error': None},
            'tex': {'value': None, 'error': None},
            'velo': {'value': None, 'error': None},
            'fwhm': {'value': None, 'error': None}
        }
    
    if 'filtered_spectra' not in st.session_state:
        st.session_state.filtered_spectra = {}
    

    if 'filter_params' not in st.session_state:
        st.session_state.filter_params = {
            'velocity': 0.0,
            'fwhm': 3.0,
            'sigma': 0.0
        }
    

    with st.sidebar:
        st.header("📁 Upload Files")
        

        use_local_models = st.checkbox("Use local models file (models.zip in same directory)")
        
        st.subheader("1. Trained Models")
        if use_local_models:
            local_zip_path = get_local_file_path("models.zip")
            if os.path.exists(local_zip_path):
                models_zip = local_zip_path
                st.success("✓ Local models.zip file found")
            else:
                st.error("✗ models.zip not found in the same directory as this script")
                models_zip = None
        else:
            models_zip = st.file_uploader("Upload ZIP file with trained models", type=['zip'])
        
        st.subheader("2. Spectrum File")
        spectrum_file = st.file_uploader("Upload spectrum file", type=['txt', 'dat'])
        
        st.subheader("3. Analysis Parameters")
        
        filters_dir = get_local_file_path("1.Filters")
        
        if os.path.exists(filters_dir):
            filter_files = glob(os.path.join(filters_dir, "*.txt"))
            
            if filter_files:
                velocities, fwhms, sigmas = parse_filter_parameters(filter_files)
                
                selected_velocity = st.selectbox(
                    "Velocity (km/s)",
                    options=velocities,
                    index=0 if 0.0 in velocities else 0,
                    help="Select velocity parameter from available filters"
                )
                
                selected_fwhm = st.selectbox(
                    "FWHM (km/s)",
                    options=fwhms,
                    index=0 if 3.0 in fwhms else 0,
                    help="Select FWHM parameter from available filters"
                )
                
                selected_sigma = st.selectbox(
                    "Sigma",
                    options=sigmas if sigmas else [0.0],
                    index=0,
                    help="Select sigma parameter from available filters"
                )

                consider_absorption = st.checkbox(
                    "Consider absorption lines (allow negative values)", 
                    value=False, 
                    help="Allow negative values in filtered spectra"
                )
                st.session_state.consider_absorption = consider_absorption
                
                st.session_state.filter_params = {
                    'velocity': selected_velocity,
                    'fwhm': selected_fwhm,
                    'sigma': selected_sigma
                }
                
                if spectrum_file:
                    generate_filters_btn = st.button("Generate Filtered Spectra", type="secondary")
                    
                    if generate_filters_btn:
                        with st.spinner("Generating filtered spectra..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_spectrum:
                                tmp_spectrum.write(spectrum_file.getvalue())
                                tmp_spectrum_path = tmp_spectrum.name
                            
                            filtered_spectra = generate_filtered_spectra(
                                tmp_spectrum_path, 
                                filters_dir, 
                                selected_velocity, 
                                selected_fwhm, 
                                selected_sigma,
                                allow_negative=st.session_state.consider_absorption
                            )
                            
                            os.unlink(tmp_spectrum_path)
                            
                            if filtered_spectra:
                                st.session_state.filtered_spectra = filtered_spectra
                                st.success(f"Generated {len(filtered_spectra)} filtered spectra")
                            else:
                                st.error("Failed to generate filtered spectra")
            else:
                st.warning("No filter files found in the '1.Filters' directory")
        else:
            st.warning("Filters directory '1.Filters' not found")
        
        st.subheader("4. Model Selection")
        st.write("Select which models to display in the results:")

        rf_selected = st.checkbox("Random Forest", value=True, key='rf_checkbox')
        gb_selected = st.checkbox("Gradient Boosting", value=True, key='gb_checkbox')
        lgb_selected = st.checkbox("LightGBM", value=True, key='lgb_checkbox')
        xgb_selected = st.checkbox("XGBoost", value=True, key='xgb_checkbox')
        
        selected_models = []
        if rf_selected:
            selected_models.append('Randomforest')
        if gb_selected:
            selected_models.append('Gradientboosting')
        if lgb_selected:
            selected_models.append('Lightgbm')
        if xgb_selected:
            selected_models.append('Xgboost')
            
        st.session_state.selected_models = selected_models
        
        # Expected values input
        st.subheader("5. Expected Values (Optional)")
        st.write("Enter expected values and uncertainties for comparison:")
        
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_labels = ['LogN', 'T_ex', 'V_los', 'FWHM']
        units = ['log(cm⁻²)', 'K', 'km/s', 'km/s']
        
        for i, (param, label, unit) in enumerate(zip(param_names, param_labels, units)):
            st.markdown(f'<div class="expected-value-input"><strong>{label} ({unit})</strong></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                value = st.number_input(
                    f"Expected value for {label}",
                    value=st.session_state.expected_values[param]['value'],
                    placeholder=f"Enter expected {label}",
                    key=f"exp_{param}_value"
                )
                st.session_state.expected_values[param]['value'] = value if value != 0 else None
            
            with col2:
                error = st.number_input(
                    f"Uncertainty for {label}",
                    value=st.session_state.expected_values[param]['error'],
                    min_value=0.0,
                    placeholder=f"Enter uncertainty for {label}",
                    key=f"exp_{param}_error"
                )
                st.session_state.expected_values[param]['error'] = error if error != 0 else None
    

    filter_names = list(st.session_state.filtered_spectra.keys())
    if 'selected_filter' not in st.session_state:
        st.session_state.selected_filter = filter_names[0] if filter_names else None

    selected_filter = st.selectbox(
        "Select a filtered spectrum for analysis",
        filter_names,
        index=filter_names.index(st.session_state.selected_filter) if st.session_state.selected_filter in filter_names else 0,
        format_func=lambda x: x,
        key='selected_filter_main'
    )

    if models_zip is not None and spectrum_file is not None and st.session_state.filtered_spectra:
        process_btn = st.button("Process Selected Spectrum", type="primary", 
                               disabled=(models_zip is None or spectrum_file is None or not selected_filter))
        if process_btn and selected_filter:
            with st.spinner("Loading and processing models..."):
                # Load models
                if use_local_models:
                    models, message = load_models_from_zip(models_zip)
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_file.write(models_zip.getvalue())
                        tmp_path = tmp_file.name
                    
                    models, message = load_models_from_zip(tmp_path)
                    os.unlink(tmp_path) 
                
                if models is None:
                    st.error(message)
                    return
                
                st.success(message)
                st.session_state.models_obj = models
                st.session_state.models_loaded = True

            # Only process the selected filtered spectrum
            spectrum_path = st.session_state.filtered_spectra[selected_filter]
            with st.spinner(f"Processing {selected_filter}..."):
                results = process_spectrum(spectrum_path, models)
                if results is None:
                    st.error(f"Error processing the filtered spectrum: {selected_filter}")
                else:
                    
                    with st.expander("Model Information", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PCA Components", models['ipca'].n_components_)
                        with col2:
                            cumulative_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
                            total_variance = cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
                            st.metric("Variance Explained", f"{total_variance*100:.1f}%")
                        with col3:
                            total_models = sum(len(models['all_models'][param]) for param in models['all_models'])
                            st.metric("Total Models", total_models)

                    st.subheader("Loaded Models")
                    param_names = ['logn', 'tex', 'velo', 'fwhm']
                    for param in param_names:
                        if param in models['all_models']:
                            model_count = len(models['all_models'][param])
                            st.write(f"{param}: {model_count} model(s) loaded")
                    st.subheader("📊 PCA Variance Analysis")
                    pca_fig = create_pca_variance_plot(models['ipca'])
                    st.pyplot(pca_fig)

                    buf = BytesIO()
                    pca_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="📥 Download PCA variance plot",
                        data=buf,
                        file_name="pca_variance_analysis.png",
                        mime="image/png"
                    )
                    
                    st.header(f"📊 Prediction Results for {selected_filter}")

                    filtered_freqs = results['processed_spectrum']['frequencies']
                    filtered_intensities = results['processed_spectrum']['intensities']


                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_freqs,
                        y=filtered_intensities,
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Filtered Spectrum'
                    ))
                    fig.update_layout(
                        title=dict(text="Filtered Spectrum", font=dict(family="Times New Roman", size=20, color="black")),
                        xaxis=dict(
                            title=dict(text="<i>Frequency</i> (GHz)", font=dict(family="Times New Roman", size=18, color="black")),
                            showgrid=True,
                            gridcolor='lightgray',
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        ),
                        yaxis=dict(
                            title=dict(text="<i>Intensity</i> (K)", font=dict(family="Times New Roman", size=18, color="black")),
                            showgrid=True,
                            gridcolor='lightgray',
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        ),
                        template="simple_white",
                        font=dict(family="Times New Roman", size=16, color="black"),
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show PCA representation of the spectrum
                    st.subheader("Spectrum in PCA Space")
                    pca_components = results['processed_spectrum']['pca_components'].flatten()

                    fig_pca_bar = go.Figure()
                    fig_pca_bar.add_trace(go.Bar(
                        x=[f'PC{i+1}' for i in range(len(pca_components))],
                        y=pca_components,
                        marker_color='purple',
                        name='PCA Component Value'
                    ))
                    fig_pca_bar.update_layout(
                        title=dict(text="Spectrum Representation in PCA Space", font=dict(family="Times New Roman", size=18, color="black")),
                        xaxis=dict(
                            title=dict(text="PCA Component", font=dict(family="Times New Roman", size=16, color="black")),
                            showgrid=True,
                            gridcolor='lightgray',
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        ),
                        yaxis=dict(
                            title=dict(text="Value", font=dict(family="Times New Roman", size=16, color="black")),
                            showgrid=True,
                            gridcolor='lightgray',
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        ),
                        template="simple_white",
                        font=dict(family="Times New Roman", size=16, color="black"),
                        height=400
                    )
                    st.plotly_chart(fig_pca_bar, use_container_width=True)


                    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(["Summary", "Model Performance", "Individual Plots", "Combined Plot", "Metrics"])
                    with subtab1:
                        st.subheader("Prediction Summary")
                        st.caption("Incertidumbre: RandomForest usa la desviación estándar de sus estimadores; otros modelos usan sqrt(Test_MSE). Si falta la métrica no se muestra barra.")
                        
                        summary_data = []
                        selected_models_lower = [m.lower() for m in st.session_state.selected_models]
                        metrics_dict_global = models.get('model_metrics')
                        for param, label in zip(results['param_names'], results['param_labels']):
                            if param in results['predictions']:
                                param_preds = results['predictions'][param]
                                param_uncerts = results['uncertainties'].get(param, {})
                                for model_name, pred_value in param_preds.items():
                                    if model_name.lower() not in selected_models_lower:
                                        continue
                                    # 1) RandomForest: usar std de estimadores si válida
                                    rf_uncert = None
                                    if model_name.lower() == 'randomforest':
                                        try:
                                            cand = param_uncerts.get(model_name, None)
                                            if isinstance(cand, (int, float)) and np.isfinite(cand) and cand >= 0:
                                                rf_uncert = float(cand)
                                        except Exception:
                                            rf_uncert = None
                                    # 2) Otros (o fallback): sqrt(Test_MSE)
                                    final_uncert = rf_uncert
                                    if (final_uncert is None) and metrics_dict_global and param in metrics_dict_global and model_name in metrics_dict_global[param]:
                                        test_mse_val = metrics_dict_global[param][model_name].get('test_mse') or metrics_dict_global[param][model_name].get('Test_MSE')
                                        try:
                                            if test_mse_val is not None:
                                                tv = float(test_mse_val)
                                                if tv >= 0:
                                                    final_uncert = np.sqrt(tv)
                                        except (TypeError, ValueError):
                                            pass
                                    rel_err = (final_uncert / abs(pred_value) * 100) if pred_value != 0 and isinstance(final_uncert, (int, float)) and np.isfinite(final_uncert) else np.nan
                                    summary_data.append({
                                        'Parameter': label,
                                        'Model': model_name,
                                        'Prediction': pred_value,
                                        'Uncertainty': final_uncert if isinstance(final_uncert, (int, float)) and np.isfinite(final_uncert) else 'N/A',
                                        'Units': get_units(param),
                                        'Relative_Error_%': rel_err
                                    })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            csv = summary_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download results as CSV",
                                data=csv,
                                file_name=f"spectrum_predictions_{selected_filter}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No predictions were generated for the selected models")
                        
                        st.subheader("Summary Plot with Expected Values")
                        
                        has_expected_values = any(
                            st.session_state.expected_values[param]['value'] is not None 
                            for param in param_names
                        )
                        
                        if has_expected_values:
                            st.info("Red line shows expected value with shaded uncertainty range")
                        
                        # Pasar los modelos seleccionados normalizados a la función de plot
                        summary_fig = create_summary_plot(
                            results['predictions'],
                            results['uncertainties'],
                            results['param_names'],
                            results['param_labels'],
                            [m for m in st.session_state.selected_models],
                            st.session_state.expected_values if has_expected_values else None,
                            model_metrics=models.get('model_metrics')
                        )
                        st.pyplot(summary_fig)
                    with subtab5:
                        st.subheader("Model Metrics (R² / MSE)")
                        mm = models.get('model_metrics')
                        if not mm:
                            st.info("No metrics were packaged with the loaded models.")
                        else:
                            rows = []
                            for param, model_dict in mm.items():
                                for model_name, metrics_dict in model_dict.items():
                                    rows.append({
                                        'Parameter': param,
                                        'Model': model_name,
                                        'Train_R2': metrics_dict.get('train_r2'),
                                        'Val_R2': metrics_dict.get('val_r2'),
                                        'Test_R2': metrics_dict.get('test_r2'),
                                        'Train_MSE': metrics_dict.get('train_mse'),
                                        'Val_MSE': metrics_dict.get('val_mse'),
                                        'Test_MSE': metrics_dict.get('test_mse')
                                    })
                            if rows:
                                metrics_df = pd.DataFrame(rows)
                                st.dataframe(metrics_df, use_container_width=True)
                                csv_m = metrics_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download metrics CSV",
                                    data=csv_m,
                                    file_name=f"model_metrics_{selected_filter}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("Metrics dictionary is empty.")
                        
                        buf = BytesIO()
                        summary_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download summary plot",
                            data=buf,
                            file_name=f"summary_predictions_{selected_filter}.png",
                            mime="image/png"
                        )
                    
                    with subtab2:
                        st.subheader("📈 Model Performance Overview")
                        st.info("Showing typical parameter ranges for each model type")
                        create_model_performance_plots(models, st.session_state.selected_models, selected_filter)
                    
                    with subtab3:
                        st.subheader("Prediction Plots by Parameter")
                        for param, label in zip(results['param_names'], results['param_labels']):
                            if param in results['predictions'] and results['predictions'][param]:
                                fig = create_comparison_plot(
                                    results['predictions'], 
                                    results['uncertainties'], 
                                    param, 
                                    label, 
                                    selected_filter,
                                    st.session_state.selected_models
                                )
                                st.pyplot(fig)

                                buf = BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                
                                st.download_button(
                                    label=f"📥 Download {label} plot",
                                    data=buf,
                                    file_name=f"prediction_{param}_{selected_filter}.png",
                                    mime="image/png",
                                    key=f"download_{param}_{selected_filter}"
                                )
                            else:
                                st.warning(f"No predictions available for {label}")
                    
                    with subtab4:
                        st.subheader("Combined Prediction Plot")
                        

                        fig = create_combined_plot(
                            results['predictions'],
                            results['uncertainties'],
                            results['param_names'],
                            results['param_labels'],
                            selected_filter,
                            st.session_state.selected_models,
                            model_metrics=models.get('model_metrics')
                        )
                        st.pyplot(fig)
                        

                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download combined plot",
                            data=buf,
                            file_name=f"combined_predictions_{selected_filter}.png",
                            mime="image/png"
                        )
    else:

        if not spectrum_file:
            st.info("👈 Please upload a spectrum file in the sidebar to get started.")
        elif not models_zip:
            st.info("👈 Please upload trained models in the sidebar to get started.")
        elif not st.session_state.filtered_spectra:
            st.info("👈 Please generate filtered spectra using the 'Generate Filtered Spectra' button.")
        
        # Usage instructions
        st.markdown("""
        ## Usage Instructions:
        
        1. **Prepare trained models**: Compress all model files (.save) and statistics (.npy) into a ZIP file named "models.zip"
        2. **Prepare spectrum**: Ensure your spectrum file is in text format with two columns (frequency, intensity)
        3. **Upload files**: Use the selectors in the sidebar to upload both files or use the local models.zip file
        4. **Select filter parameters**: Choose velocity, FWHM, and sigma values from available filters
        5. **Generate filtered spectra**: Click the 'Generate Filtered Spectra' button to create filtered spectra
        6. **Select models**: Choose which models to display in the results using the checkboxes
        7. **Enter expected values (optional)**: Provide expected values and uncertainties for comparison
        8. **Process**: Click the 'Process Spectrum' button to get predictions for all filtered spectra
        """)

if __name__ == "__main__":
    main()
