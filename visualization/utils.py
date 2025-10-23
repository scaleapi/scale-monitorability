"""
Shared utilities for the visualization module.
This module contains common functions, constants, and configurations used across
streamlit_app.py, overview.py, and plotting.ipynb.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Dict, Any, Optional
import glob

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# File paths
RESULTS_DIR = Path("../results")

# File filtering
EXCLUDED_FILES = ['perturbation_data.jsonl']

# UI options
INCONSISTENCY_OPTIONS = ["All", "Inconsistent (True)", "Consistent (False)"]
BOOLEAN_CONVERSION_OPTIONS = {
    "Only Inconsistent": ["Inconsistent"],
    "Inconsistent + Ambiguous": ["Inconsistent", "Ambiguous"]
}
PERFORMANCE_CATEGORIES = ["All", "True Positives", "False Positives", "True Negatives", "False Negatives"]

# Grader evaluation
# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

def setup_page_config(title: str):
    """Set up Streamlit page configuration with common settings."""
    st.set_page_config(
        page_title=title,
        layout="wide",
        initial_sidebar_state="expanded"
    )

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_jsonl(file_path: str, _last_modified: Optional[float] = None) -> Optional[pd.DataFrame]:
    """Load JSONL data and convert to DataFrame.
    
    Args:
        file_path: Path to the JSONL file
        _last_modified: Optional last modified time for caching (Streamlit specific)
        
    Returns:
        DataFrame containing the loaded data, or None if error
    """
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)
    except Exception as e:
        if 'streamlit' in globals():
            st.error(f"Error loading {file_path}: {e}")
        else:
            print(f"Error loading {file_path}: {e}")
        return None

def discover_jsonl_files() -> List[Path]:
    """Discover and return sorted JSONL files from results directory."""
    jsonl_files = []
    
    if RESULTS_DIR.exists():
        jsonl_files.extend(RESULTS_DIR.rglob("*.jsonl"))
    
    # Filter out excluded files
    jsonl_files = [file for file in jsonl_files if file.name not in EXCLUDED_FILES]
    
    # Sort by modification time (newest first)
    jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return jsonl_files

@st.cache_data
def get_grader_info():
    """Extract grader information from graders.py"""
    try:
        from graders import ALL_FAITHFULNESS_GRADER_PROMPTS
        
        # Now we just have two graders: simulatability and tuned simulatability
        grader_names = ['faithfulness_simulability', 'faithfulness_simulability_tuned']
        
        # Get available categories
        category_specific_graders = ALL_FAITHFULNESS_GRADER_PROMPTS['faithfulness_simulability_tuned']
        
        # Create category-based grader mapping - all categories use the same two graders
        category_graders = {}
        for category in category_specific_graders.keys():
            category_graders[category] = [
                'faithfulness_simulability',      # Simulatability grader (same for all)
                'faithfulness_simulability_tuned'   # Tuned simulatability grader
            ]
        
        return {
            'all_graders': {name: name for name in grader_names},  # Simple mapping for compatibility
            'category_graders': category_graders,
            'grader_names': grader_names
        }
    except ImportError as e:
        if 'streamlit' in globals():
            st.error(f"Error importing graders: {e}")
        return {
            'all_graders': {},
            'category_graders': {},
            'grader_names': []
        }

@st.cache_data
def discover_model_files() -> Dict[str, Dict[str, List[Path]]]:
    """Discover all model result files in results directory."""
    if not RESULTS_DIR.exists():
        return {}
    
    model_files = {}
    
    # Find all JSONL files recursively in the results directory
    for file_path in RESULTS_DIR.rglob("*.jsonl"):
        # Skip if it's the original dataset file
        if file_path.name == 'perturbation_data.jsonl':
            continue
            
        # Skip files that can't be accessed
        try:
            file_path.stat()  # Test if file is accessible
        except (OSError, PermissionError):
            continue
            
        # Get the model name from the directory structure
        relative_path = file_path.relative_to(RESULTS_DIR)
        model_path_parts = relative_path.parts[:-1]  # Remove the filename
        
        if len(model_path_parts) == 0:
            continue  # Skip files directly in results directory
            
        # Join all directory parts to form the model name
        model_name = "/".join(model_path_parts)
        
        if model_name not in model_files:
            model_files[model_name] = {}
        
        # Extract dataset name from filename
        filename = file_path.stem
        if "_" in filename:
            dataset_name = filename.split("_")[0]
            
            # Group files by dataset
            if dataset_name not in model_files[model_name]:
                model_files[model_name][dataset_name] = []
            
            model_files[model_name][dataset_name].append(file_path)
    
    # Sort files by modification time for each dataset
    for model_name in model_files:
        for dataset_name in model_files[model_name]:
            model_files[model_name][dataset_name].sort(
                key=lambda f: f.stat().st_mtime, reverse=True
            )
    
    return model_files

# =============================================================================
# DATA CONVERSION AND PROCESSING
# =============================================================================

def convert_simulatability_to_bool(value, true_values: List[str]) -> bool:
    """Convert simulatability_pred values to boolean based on selected criteria.
    
    Args:
        value: The raw prediction value ('Consistent', 'Inconsistent', 'Ambiguous', or unparsed text)
        true_values: List of values to consider as True (unfaithful)
    
    Returns:
        bool: True if unfaithful, False if faithful
    """
    if pd.isna(value):
        return False
    
    # Handle string values (extract from longer text if needed)
    if isinstance(value, str):
        value = value.strip()
        
        # Check if it's a clean value first
        if value in ['Consistent', 'Inconsistent', 'Ambiguous']:
            return value in true_values
        
        # If not clean, try to extract from longer text
        # Look for the key words in the text
        if 'Consistent' in value and 'Inconsistent' not in value:
            return 'Consistent' in true_values
        elif 'Inconsistent' in value:
            return 'Inconsistent' in true_values
        elif 'Ambiguous' in value:
            return 'Ambiguous' in true_values
    
    # Fallback: try direct comparison
    return value in true_values

def apply_boolean_conversion(df: pd.DataFrame, grader_name: str, true_values: List[str]) -> pd.Series:
    """Apply boolean conversion to a specific grader column.
    
    Args:
        df: DataFrame containing the grader predictions
        grader_name: Name of the grader
        true_values: List of values to consider as True
        
    Returns:
        Boolean series indicating unfaithful predictions
    """
    pred_col = f"{grader_name}_pred"
    if pred_col in df.columns:
        if grader_name in ['faithfulness_simulability', 'faithfulness_simulability_tuned']:
            return df[pred_col].apply(lambda x: convert_simulatability_to_bool(x, true_values))
        else:
            return df[pred_col].fillna(False).astype(bool)
    else:
        return pd.Series([False] * len(df), index=df.index)

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_inconsistency_rate(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate inconsistency rate for a dataset."""
    total_cases = len(df)
    if total_cases == 0:
        return {'inconsistency_rate': 0.0, 'num_inconsistent': 0, 'num_total': 0}
    
    # Count inconsistent cases
    inconsistent_mask = df['inconsistency_label'].fillna(False).astype(bool) == True
    num_inconsistent = inconsistent_mask.sum()
    
    inconsistency_rate = num_inconsistent / total_cases if total_cases > 0 else 0.0
    
    # Calculate confidence interval using Wilson score interval
    if total_cases > 0:
        p = inconsistency_rate
        n = total_cases
        if n > 0 and 0 < p < 1:
            z = 1.96  # 95% confidence
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n)) / denominator
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
        else:
            ci_lower = ci_upper = p
        confidence_interval = (ci_lower, ci_upper)
    else:
        confidence_interval = (0.0, 0.0)
    
    return {
        'inconsistency_rate': inconsistency_rate,
        'num_inconsistent': num_inconsistent,
        'num_total': total_cases,
        'confidence_interval': confidence_interval
    }

def calculate_baseline_inconsistency_rate(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate baseline inconsistency rate for a dataset using baseline_inconsistency_label."""
    total_cases = len(df)
    if total_cases == 0:
        return {'baseline_inconsistency_rate': 0.0, 'num_baseline_inconsistent': 0, 'num_total': 0}
    
    # Count baseline inconsistent cases
    if 'baseline_inconsistency_label' in df.columns:
        baseline_inconsistent_mask = df['baseline_inconsistency_label'].fillna(False).astype(bool) == True
        num_baseline_inconsistent = baseline_inconsistent_mask.sum()
    else:
        num_baseline_inconsistent = 0
    
    baseline_inconsistency_rate = num_baseline_inconsistent / total_cases if total_cases > 0 else 0.0
    
    # Calculate confidence interval
    if total_cases > 0:
        p = baseline_inconsistency_rate
        n = total_cases
        if n > 0 and 0 < p < 1:
            z = 1.96  # 95% confidence
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n)) / denominator
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
        else:
            ci_lower = ci_upper = p
        confidence_interval = (ci_lower, ci_upper)
    else:
        confidence_interval = (0.0, 0.0)
    
    return {
        'baseline_inconsistency_rate': baseline_inconsistency_rate,
        'num_baseline_inconsistent': num_baseline_inconsistent,
        'num_total': total_cases,
        'confidence_interval': confidence_interval
    }

def calculate_verbalization_score(df: pd.DataFrame, grader_names: List[str], method: str = 'any', 
                                 grader_true_values: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate verbalization score using specified graders and method.
    
    Args:
        df: DataFrame containing the data
        grader_names: List of grader names to use
        method: 'any', 'all', or specific grader name
        grader_true_values: List of values to consider True for grader predictions
        
    Returns:
        Dictionary containing verbalization metrics
    """
    if grader_true_values is None:
        grader_true_values = GRADER_TRUE_VALUES
    
    # Filter to only inconsistent cases
    inconsistent_mask = df['inconsistency_label'].fillna(False).astype(bool) == True
    inconsistent_df = df[inconsistent_mask]
    
    if len(inconsistent_df) == 0:
        return {
            'verbalization_score': 0.0,
            'num_verbalized': 0,
            'num_inconsistent': 0,
            'verbalization_rate': 0.0
        }
    
    verbalization_detected = pd.Series([False] * len(inconsistent_df), index=inconsistent_df.index)
    
    # Apply the specified method
    if method == 'any':
        # At least one grader must detect verbalization
        for grader_name in grader_names:
            verbalization_detected |= apply_boolean_conversion(inconsistent_df, grader_name, grader_true_values)
    elif method == 'all':
        # All graders must detect verbalization
        verbalization_detected = pd.Series([True] * len(inconsistent_df), index=inconsistent_df.index)
        for grader_name in grader_names:
            verbalization_detected &= apply_boolean_conversion(inconsistent_df, grader_name, grader_true_values)
    else:
        # Specific grader
        verbalization_detected = apply_boolean_conversion(inconsistent_df, method, grader_true_values)
    
    num_verbalized = verbalization_detected.sum()
    num_inconsistent = len(inconsistent_df)
    verbalization_rate = num_verbalized / num_inconsistent if num_inconsistent > 0 else 0.0
    
    return {
        'verbalization_score': verbalization_rate,
        'num_verbalized': num_verbalized,
        'num_inconsistent': num_inconsistent,
        'verbalization_rate': verbalization_rate
    }

def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard classification performance metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division="warn"),
        'recall': recall_score(y_true, y_pred, zero_division="warn"),
        'f1': f1_score(y_true, y_pred, zero_division="warn")
    }

def calculate_grader_performance(df: pd.DataFrame, grader_name: str, category: Optional[str] = None, 
                               grader_true_values: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate grader performance metrics against ground truth.
    
    Args:
        df: DataFrame containing the data
        grader_name: Name of the grader to evaluate
        category: Optional category to filter by
        grader_true_values: List of values to consider as True for grader predictions
        
    Returns:
        Dictionary containing performance metrics
    """
    if grader_true_values is None:
        grader_true_values = GRADER_TRUE_VALUES
    
    # Filter by category if specified
    if category and category != "All":
        df = df[df['category'] == category]
    
    if len(df) == 0:
        return {'error': 'No data available for the selected filters'}
    
    # Get ground truth (inconsistency labels)
    y_true = df['inconsistency_label'].fillna(False).astype(bool)
    
    # Get grader predictions
    y_pred = apply_boolean_conversion(df, grader_name, grader_true_values)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(y_true.values, y_pred.values)
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_samples': len(df)
        })
    else:
        # Handle edge case where confusion matrix is not 2x2
        metrics.update({
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_samples': len(df)
        })
    
    return metrics

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_available_graders(df: pd.DataFrame) -> List[str]:
    """Get list of available graders from DataFrame columns."""
    grader_columns = [col for col in df.columns if col.endswith('_pred')]
    return [col.replace('_pred', '') for col in grader_columns]

def filter_excluded_categories(df: pd.DataFrame, exclude_categories: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter out excluded categories from DataFrame."""
    if exclude_categories is None:
        exclude_categories = EXCLUDED_CATEGORIES
    
    if exclude_categories:
        return df[~df['category'].isin(exclude_categories)]
    return df 