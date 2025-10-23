import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import (
    # Configuration and constants
    RESULTS_DIR, GRADER_TRUE_VALUES,
    # Page setup
    setup_page_config,
    # Data loading and processing
    load_jsonl, discover_model_files,
    # Data conversion
    convert_simulatability_to_bool, apply_boolean_conversion,
    # Calculation functions
    calculate_inconsistency_rate, calculate_baseline_inconsistency_rate,
    calculate_verbalization_score
)

# Set page configuration
setup_page_config("Model Verbalization Comparison")

def calculate_model_scores(model_data, grader_method, grader_true_values=None):
    """Calculate overall model scores and category breakdown"""
    grader_names = get_grader_names_from_data(model_data)
    
    # Calculate overall metrics
    overall_metrics = calculate_verbalization_score(model_data, grader_names, grader_method, grader_true_values)
    overall_inconsistency = calculate_inconsistency_rate(model_data)
    overall_baseline_inconsistency = calculate_baseline_inconsistency_rate(model_data)
    
    # Combine overall metrics
    overall_scores = {**overall_metrics}
    overall_scores.update({f'overall_{k}': v for k, v in overall_inconsistency.items()})
    overall_inconsistency['inconsistency_confidence_interval'] = overall_inconsistency.pop('confidence_interval')
    overall_baseline_inconsistency['baseline_inconsistency_confidence_interval'] = overall_baseline_inconsistency.pop('confidence_interval')
    overall_scores.update(overall_baseline_inconsistency)
    
    # Calculate category breakdown
    category_scores = {}
    for category in model_data['category'].unique():
        if pd.isna(category):
            continue
        cat_data = model_data[model_data['category'] == category]
        cat_grader_names = get_grader_names_from_data(cat_data)
        
        cat_metrics = calculate_verbalization_score(cat_data, cat_grader_names, grader_method, grader_true_values)
        cat_inconsistency = calculate_inconsistency_rate(cat_data)
        cat_baseline_inconsistency = calculate_baseline_inconsistency_rate(cat_data)
        
        category_scores[category] = {
            **cat_metrics,
            **{f'category_{k}': v for k, v in cat_inconsistency.items()},
            **{f'category_{k}': v for k, v in cat_baseline_inconsistency.items()}
        }
    
    return overall_scores, category_scores

@st.cache_data
def get_grader_names_from_data(df):
    """Extract available grader names from dataframe columns"""
    grader_names = []
    for col in df.columns:
        if col.endswith('_pred') and col.startswith('faithfulness_'):
            grader_name = col[:-5]  # Remove '_pred' suffix
            grader_names.append(grader_name)
    return grader_names

# calculate_verbalization_score is imported from utils

# calculate_inconsistency_rate and calculate_baseline_inconsistency_rate are imported from utils

# Second calculate_model_scores function removed - using the one defined above

def main():
    st.title("Model Verbalization Comparison")
    st.markdown("Compare verbalization detection performance across different models")
    st.markdown("---")
    
    # Discover available model files
    model_files = discover_model_files()
    
    if not model_files:
        st.error("No model result files found in the results directory.")
        st.info("Expected structure: results/{model_name}/{dataset_name}_{timestamp}_{config}.jsonl")
        st.info("Make sure the results directory exists and contains JSONL files with the expected structure.")
        return
    
    st.sidebar.header("Configuration")
    
    # Cache control
    if st.sidebar.button("🔄 Clear Cache & Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Dataset selection (get all available datasets across all models)
    all_datasets = set()
    for model_data in model_files.values():
        all_datasets.update(model_data.keys())
    
    if not all_datasets:
        st.error("No datasets found in model files.")
        st.info("Make sure your JSONL files follow the naming convention: {dataset_name}_{timestamp}_{config}.jsonl")
        return
    
    selected_dataset = st.sidebar.selectbox(
        "Select dataset:",
        sorted(all_datasets),
        help="Dataset to analyze across models"
    )
    
    # Simulatability Boolean Conversion
    st.sidebar.subheader("Faithfulness Boolean Conversion")
    
    boolean_conversion_options = {
        "Only Inconsistent": ["Inconsistent"],
        "Inconsistent + Ambiguous": ["Inconsistent", "Ambiguous"]
    }
    
    selected_conversion = st.sidebar.selectbox(
        "What counts as True (unfaithful) for grader predictions?",
        options=list(boolean_conversion_options.keys()),
        index=0,
        help="Select which values should be considered True (unfaithful) for grader predictions"
    )
    
    grader_true_values = boolean_conversion_options[selected_conversion]
    
    # Model selection (only show models that have the selected dataset)
    available_models = [
        model_name for model_name, datasets in model_files.items()
        if selected_dataset in datasets
    ]
    
    if not available_models:
        st.error(f"No models found with dataset '{selected_dataset}'.")
        return
    
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        available_models,
        default=[]  # No default selection
    )
    
    if not selected_models:
        st.warning("Please select at least one model to analyze.")
        return
    
    # Grader method selection
    grader_method = st.sidebar.selectbox(
        "Verbalization detection method:",
        ["any", "all", "faithfulness_simulability", "faithfulness_simulability_tuned"],
        index=0,
        help="'any': Any grader detects verbalization, 'all': All graders must detect, or select specific grader"
    )
    
    # Load Data Button
    load_data = st.sidebar.button("Load Data", type="primary", use_container_width=True)
    
    # Initialize session state for data persistence
    if 'model_scores' not in st.session_state:
        st.session_state.model_scores = {}
    if 'model_data_cache' not in st.session_state:
        st.session_state.model_data_cache = {}
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = []
    if 'current_grader_method' not in st.session_state:
        st.session_state.current_grader_method = None
    
    # Check if we need to reload data (different models selected or different grader method)
    models_changed = set(selected_models) != set(st.session_state.loaded_models)
    grader_changed = grader_method != st.session_state.current_grader_method
    
    # Load and process data when button is clicked or when models/grader method changes
    if load_data or (models_changed and st.session_state.loaded_models) or (grader_changed and st.session_state.loaded_models):
        with st.spinner("Loading and processing data..."):
            st.session_state.model_scores = {}
            st.session_state.model_data_cache = {}
            
            for model in selected_models:
                file_path = model_files[model][selected_dataset]
                try:
                    file_path_obj = Path(file_path)
                    file_mtime = file_path_obj.stat().st_mtime
                    df = load_jsonl(file_path, _last_modified=file_mtime)
                    
                    if df is not None and len(df) > 0:
                        st.session_state.model_data_cache[model] = df
                        st.session_state.model_scores[model] = calculate_model_scores(df, grader_method, grader_true_values)
                        st.sidebar.success(f"✓ Loaded {model} on {selected_dataset}: {len(df)} records")
                    else:
                        st.sidebar.error(f"✗ Failed to load {model} on {selected_dataset}: No data found")
                except Exception as e:
                    st.sidebar.error(f"✗ Failed to load {model} on {selected_dataset}: {str(e)}")
            
            st.session_state.loaded_models = selected_models.copy()
            st.session_state.current_grader_method = grader_method
    
    # Check if data is loaded
    if not st.session_state.model_scores:
        st.info("👆 Please select models and click 'Load Data' to begin analysis.")
        return
    
    # Use session state data
    model_scores = st.session_state.model_scores
    model_data_cache = st.session_state.model_data_cache
    loaded_models = st.session_state.loaded_models
    
    # Show info about loaded models
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Currently Loaded Models:**")
    for model in loaded_models:
        st.sidebar.markdown(f"• {model}")
    
    # Main analysis tabs
    tab1, tab2, tab3 = st.tabs(["Overall Comparison", "Category Breakdown", "Inconsistency Analysis"])
    
    with tab1:
        st.header("Overall Verbalization Scores")
        
        # Aggregate options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Weighted Average (by category sizes)")
            
            weighted_scores = []
            for model in loaded_models:
                if model in model_scores:
                    # Calculate weighted average across categories
                    total_weight = 0
                    weighted_sum = 0
                    
                    for category, metrics in model_scores[model].items():
                        if category != 'Overall':
                            weight = metrics['num_total_inconsistent']
                            weighted_sum += metrics['verbalization_score'] * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        weighted_avg = weighted_sum / total_weight
                        # Calculate weighted confidence interval
                        total_successes = sum(metrics['num_verbalize_given_inconsistent'] 
                                            for category, metrics in model_scores[model].items() 
                                            if category != 'Overall')
                        ci_lower = weighted_avg - 1.96 * np.sqrt(weighted_avg * (1 - weighted_avg) / total_weight)
                        ci_upper = weighted_avg + 1.96 * np.sqrt(weighted_avg * (1 - weighted_avg) / total_weight)
                        
                        weighted_scores.append({
                            'Model': model,
                            'Verbalization Score': weighted_avg,
                            'CI Lower': max(0, ci_lower),
                            'CI Upper': min(1, ci_upper),
                            'Total Inconsistent': total_weight
                        })
            
            if weighted_scores:
                weighted_df = pd.DataFrame(weighted_scores)
                
                # Create bar chart with error bars
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=weighted_df['Model'],
                    y=weighted_df['Verbalization Score'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        arrayminus=weighted_df['Verbalization Score'] - weighted_df['CI Lower'],
                        array=weighted_df['CI Upper'] - weighted_df['Verbalization Score']
                    ),
                    text=[f"{score:.1%}" for score in weighted_df['Verbalization Score']],
                    textposition='auto',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Weighted Average Verbalization Scores",
                    xaxis_title="Model",
                    yaxis_title="Verbalization Score",
                    yaxis=dict(range=[0, 1], tickformat='.0%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(weighted_df.round(3), use_container_width=True)
        
        with col2:
            st.subheader("Unweighted Average (equal category weights)")
            
            unweighted_scores = []
            for model in loaded_models:
                if model in model_scores:
                    # Calculate unweighted average across categories
                    category_scores = [metrics['verbalization_score'] 
                                     for category, metrics in model_scores[model].items() 
                                     if category != 'Overall']
                    
                    if category_scores:
                        unweighted_avg = np.mean(category_scores)
                        unweighted_std = np.std(category_scores) / np.sqrt(len(category_scores))  # Standard error
                        
                        unweighted_scores.append({
                            'Model': model,
                            'Verbalization Score': unweighted_avg,
                            'CI Lower': max(0, unweighted_avg - 1.96 * unweighted_std),
                            'CI Upper': min(1, unweighted_avg + 1.96 * unweighted_std),
                            'Categories': len(category_scores)
                        })
            
            if unweighted_scores:
                unweighted_df = pd.DataFrame(unweighted_scores)
                
                # Create bar chart with error bars
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=unweighted_df['Model'],
                    y=unweighted_df['Verbalization Score'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        arrayminus=unweighted_df['Verbalization Score'] - unweighted_df['CI Lower'],
                        array=unweighted_df['CI Upper'] - unweighted_df['Verbalization Score']
                    ),
                    text=[f"{score:.1%}" for score in unweighted_df['Verbalization Score']],
                    textposition='auto',
                    marker_color='lightcoral'
                ))
                
                fig.update_layout(
                    title="Unweighted Average Verbalization Scores",
                    xaxis_title="Model",
                    yaxis_title="Verbalization Score",
                    yaxis=dict(range=[0, 1], tickformat='.0%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(unweighted_df.round(3), use_container_width=True)
    
    with tab2:
        st.header("Category Breakdown")
        
        # Get all categories across all models
        all_categories = set()
        for model_score in model_scores.values():
            all_categories.update(category for category in model_score.keys() if category != 'Overall')
        all_categories = sorted(all_categories)
        
        if all_categories:
            # Category selector
            selected_categories = st.multiselect(
                "Select categories to display:",
                all_categories,
                default=all_categories
            )
            
            if selected_categories:
                # Create comparison data
                comparison_data = []
                
                for category in selected_categories:
                    for model in loaded_models:
                        if model in model_scores and category in model_scores[model]:
                            metrics = model_scores[model][category]
                            comparison_data.append({
                                'Category': category,
                                'Model': model,
                                'Verbalization Score': metrics['verbalization_score'],
                                'CI Lower': metrics['verbalization_confidence_interval'][0],
                                'CI Upper': metrics['verbalization_confidence_interval'][1],
                                'Verbalized': metrics['num_verbalize_given_inconsistent'],
                                'Total Inconsistent': metrics['num_total_inconsistent']
                            })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Grouped bar chart - using go.Figure for better error bar control
                    fig = go.Figure()
                    
                    models = comparison_df['Model'].unique()
                    categories = comparison_df['Category'].unique()
                    
                    for i, model in enumerate(models):
                        model_data = comparison_df[comparison_df['Model'] == model]
                        
                        fig.add_trace(go.Bar(
                            name=model,
                            x=model_data['Category'],
                            y=model_data['Verbalization Score'],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                arrayminus=model_data['Verbalization Score'] - model_data['CI Lower'],
                                array=model_data['CI Upper'] - model_data['Verbalization Score']
                            ),
                            customdata=np.column_stack((model_data['Verbalized'], model_data['Total Inconsistent'])),
                            hovertemplate='<b>' + model + '</b><br>' +
                                        'Category: %{x}<br>' +
                                        'Verbalization Score: %{y:.1%}<br>' +
                                        'Verbalized: %{customdata[0]}<br>' +
                                        'Total Inconsistent: %{customdata[1]}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="Verbalization Scores by Category and Model",
                        xaxis_title="Category",
                        yaxis_title="Verbalization Score",
                        yaxis=dict(range=[0, 1], tickformat='.0%'),
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stacked bar chart showing inconsistent vs verbalized samples
                    st.subheader("Sample Breakdown by Category: Inconsistent (Verbalized vs Not Verbalized)")
                    
                    # Create stacked bar chart data
                    stacked_fig = go.Figure()
                    
                    # First add overlay bars showing total inconsistency rate for each model (bottom layer)
                    for i, model in enumerate(models):
                        model_data = comparison_df[comparison_df['Model'] == model]
                        
                        # Get total inconsistency rates for each category from model_scores
                        inconsistency_rates = []
                        inconsistency_ci_lower = []
                        inconsistency_ci_upper = []
                        
                        for _, row in model_data.iterrows():
                            category = row['Category']
                            if model in st.session_state.model_scores and category in st.session_state.model_scores[model]:
                                inconsistency_rates.append(st.session_state.model_scores[model][category]['inconsistency_rate'] * 100)
                                inconsistency_ci_lower.append(st.session_state.model_scores[model][category]['inconsistency_confidence_interval'][0] * 100)
                                inconsistency_ci_upper.append(st.session_state.model_scores[model][category]['inconsistency_confidence_interval'][1] * 100)
                            else:
                                inconsistency_rates.append(0)
                                inconsistency_ci_lower.append(0)
                                inconsistency_ci_upper.append(0)
                        
                        # Add semi-transparent overlay bars for total inconsistency rate
                        stacked_fig.add_trace(go.Bar(
                            name=f"{model} - Total Inconsistency Rate",
                            x=model_data['Category'],
                            y=inconsistency_rates,
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                arrayminus=np.array(inconsistency_rates) - np.array(inconsistency_ci_lower),
                                array=np.array(inconsistency_ci_upper) - np.array(inconsistency_rates)
                            ),
                            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
                            opacity=0.3,  # Semi-transparent overlay
                            offsetgroup=model,
                            legendgroup=model,
                            showlegend=False,  # Remove from legend
                            hovertemplate='<b>' + model + ' - Total Inconsistency Rate</b><br>' +
                                        'Category: %{x}<br>' +
                                        'Total Inconsistency Rate: %{y:.1f}%<extra></extra>'
                        ))
                    
                    # Then add all verbalized bars (middle layer - bottom of stack)
                    for i, model in enumerate(models):
                        model_data = comparison_df[comparison_df['Model'] == model]
                        
                        # Get total samples for each category from model_scores
                        total_samples = []
                        for _, row in model_data.iterrows():
                            category = row['Category']
                            if model in st.session_state.model_scores and category in st.session_state.model_scores[model]:
                                total_samples.append(st.session_state.model_scores[model][category]['num_total'])
                            else:
                                total_samples.append(0)
                        total_samples = np.array(total_samples)
                        
                        # Calculate percentages relative to ALL samples (not just inconsistent ones)
                        verbalized_pct_of_all = np.where(total_samples > 0, 
                                                        model_data['Verbalized'] / total_samples * 100, 0)
                        
                        stacked_fig.add_trace(go.Bar(
                            name=f"{model} - Verbalized",
                            x=model_data['Category'],
                            y=verbalized_pct_of_all,
                            customdata=np.column_stack((model_data['Verbalized'], model_data['Total Inconsistent'], 
                                                       verbalized_pct_of_all, total_samples)),
                            hovertemplate='<b>' + model + ' - Verbalized</b><br>' +
                                        'Category: %{x}<br>' +
                                        'Verbalized: %{customdata[0]} (%{customdata[2]:.1f}% of all samples)<br>' +
                                        'Total Inconsistent: %{customdata[1]}<br>' +
                                        'Total Samples: %{customdata[3]:.0f}<extra></extra>',
                            legendgroup=model,
                            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
                            opacity=0.8,
                            offsetgroup=model
                        ))
                    
                    # Then add all not verbalized bars (middle layer - top of stack)
                    for i, model in enumerate(models):
                        model_data = comparison_df[comparison_df['Model'] == model]
                        
                        # Get total samples for each category from model_scores
                        total_samples = []
                        for _, row in model_data.iterrows():
                            category = row['Category']
                            if model in st.session_state.model_scores and category in st.session_state.model_scores[model]:
                                total_samples.append(st.session_state.model_scores[model][category]['num_total'])
                            else:
                                total_samples.append(0)
                        total_samples = np.array(total_samples)
                        
                        # Calculate percentages relative to ALL samples
                        not_verbalized = model_data['Total Inconsistent'] - model_data['Verbalized']
                        not_verbalized_pct_of_all = np.where(total_samples > 0, 
                                                            not_verbalized / total_samples * 100, 0)
                        verbalized_pct_of_all = np.where(total_samples > 0, 
                                                        model_data['Verbalized'] / total_samples * 100, 0)
                        
                        stacked_fig.add_trace(go.Bar(
                            name=f"{model} - Not Verbalized",
                            x=model_data['Category'],
                            y=not_verbalized_pct_of_all,
                            base=verbalized_pct_of_all,  # Stack on top of verbalized percentage
                            customdata=np.column_stack((not_verbalized, model_data['Total Inconsistent'], 
                                                       not_verbalized_pct_of_all, total_samples)),
                            hovertemplate='<b>' + model + ' - Not Verbalized</b><br>' +
                                        'Category: %{x}<br>' +
                                        'Not Verbalized: %{customdata[0]} (%{customdata[2]:.1f}% of all samples)<br>' +
                                        'Total Inconsistent: %{customdata[1]}<br>' +
                                        'Total Samples: %{customdata[3]:.0f}<extra></extra>',
                            legendgroup=model,
                            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
                            opacity=0.4,
                            offsetgroup=model
                        ))
                    

                    
                    stacked_fig.update_layout(
                        title="Sample Breakdown: Inconsistent (Verbalized vs Not Verbalized)",
                        xaxis_title="Category",
                        yaxis_title="Percentage of All Samples",
                        yaxis=dict(range=[0, 100], tickformat='.0f', ticksuffix='%'),
                        barmode='group',  # Group bars side by side
                        legend=dict(
                            groupclick="togglegroup"
                        )
                    )
                    
                    st.plotly_chart(stacked_fig, use_container_width=True)
                    
                    # Detailed table
                    st.subheader("Detailed Category Metrics")
                    display_df = comparison_df.copy()
                    display_df['Verbalization Score'] = display_df['Verbalization Score'].apply(lambda x: f"{x:.1%}")
                    display_df['95% CI'] = display_df.apply(
                        lambda row: f"[{row['CI Lower']:.1%}, {row['CI Upper']:.1%}]", axis=1
                    )
                    display_df = display_df[['Category', 'Model', 'Verbalization Score', '95% CI', 'Verbalized', 'Total Inconsistent']]
                    st.dataframe(display_df, use_container_width=True)
    
    with tab3:
        st.header("Inconsistency Analysis")
        st.markdown("Analysis of inconsistency rates across models and categories")
        
        # Overall inconsistency comparison
        st.subheader("Overall Inconsistency Rates")
        
        # Create inconsistency comparison data
        inconsistency_data = []
        for model in loaded_models:
            if model in model_scores:
                overall_metrics = model_scores[model].get('Overall', {})
                inconsistency_data.append({
                    'Model': model,
                    'Inconsistency Rate': overall_metrics.get('inconsistency_rate', 0),
                    'CI Lower': overall_metrics.get('inconsistency_confidence_interval', (0, 0))[0],
                    'CI Upper': overall_metrics.get('inconsistency_confidence_interval', (0, 0))[1],
                    'Inconsistent Cases': overall_metrics.get('num_inconsistent', 0),
                    'Total Cases': overall_metrics.get('num_total', 0)
                })
        
        if inconsistency_data:
            inconsistency_df = pd.DataFrame(inconsistency_data)
            
            # Create bar chart for inconsistency rates
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=inconsistency_df['Model'],
                y=inconsistency_df['Inconsistency Rate'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    arrayminus=inconsistency_df['Inconsistency Rate'] - inconsistency_df['CI Lower'],
                    array=inconsistency_df['CI Upper'] - inconsistency_df['Inconsistency Rate']
                ),
                text=[f"{rate:.1%}" for rate in inconsistency_df['Inconsistency Rate']],
                textposition='auto',
                marker_color='orange',
                customdata=np.column_stack((inconsistency_df['Inconsistent Cases'], inconsistency_df['Total Cases'])),
                hovertemplate='<b>%{x}</b><br>' +
                            'Inconsistency Rate: %{y:.1%}<br>' +
                            'Inconsistent Cases: %{customdata[0]}<br>' +
                            'Total Cases: %{customdata[1]}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Inconsistency Rates by Model",
                xaxis_title="Model",
                yaxis_title="Inconsistency Rate",
                yaxis=dict(range=[0, 1], tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            display_inconsistency_df = inconsistency_df.copy()
            display_inconsistency_df['Inconsistency Rate'] = display_inconsistency_df['Inconsistency Rate'].apply(lambda x: f"{x:.1%}")
            display_inconsistency_df['95% CI'] = display_inconsistency_df.apply(
                lambda row: f"[{row['CI Lower']:.1%}, {row['CI Upper']:.1%}]", axis=1
            )
            display_inconsistency_df = display_inconsistency_df[['Model', 'Inconsistency Rate', '95% CI', 'Inconsistent Cases', 'Total Cases']]
            st.dataframe(display_inconsistency_df, use_container_width=True)
        
        # Category-specific inconsistency rates
        st.subheader("Inconsistency Rates by Category")
        
        # Get all categories
        all_categories = set()
        for model_score in model_scores.values():
            all_categories.update(category for category in model_score.keys() if category != 'Overall')
        all_categories = sorted(all_categories)
        
        if all_categories:
            # Category selector
            selected_categories = st.multiselect(
                "Select categories to display:",
                all_categories,
                default=all_categories,
                key="inconsistency_categories"
            )
            
            if selected_categories:
                # Create category inconsistency data
                category_inconsistency_data = []
                
                for category in selected_categories:
                    for model in loaded_models:
                        if model in model_scores and category in model_scores[model]:
                            metrics = model_scores[model][category]
                            category_inconsistency_data.append({
                                'Category': category,
                                'Model': model,
                                'Inconsistency Rate': metrics.get('inconsistency_rate', 0),
                                'CI Lower': metrics.get('inconsistency_confidence_interval', (0, 0))[0],
                                'CI Upper': metrics.get('inconsistency_confidence_interval', (0, 0))[1],
                                'Inconsistent Cases': metrics.get('num_inconsistent', 0),
                                'Total Cases': metrics.get('num_total', 0)
                            })
                
                if category_inconsistency_data:
                    category_inconsistency_df = pd.DataFrame(category_inconsistency_data)
                    
                    # Grouped bar chart for category inconsistency rates
                    fig = go.Figure()
                    
                    models = category_inconsistency_df['Model'].unique()
                    
                    for model in models:
                        model_data = category_inconsistency_df[category_inconsistency_df['Model'] == model]
                        
                        fig.add_trace(go.Bar(
                            name=model,
                            x=model_data['Category'],
                            y=model_data['Inconsistency Rate'],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                arrayminus=model_data['Inconsistency Rate'] - model_data['CI Lower'],
                                array=model_data['CI Upper'] - model_data['Inconsistency Rate']
                            ),
                            customdata=np.column_stack((model_data['Inconsistent Cases'], model_data['Total Cases'])),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Category: %{x}<br>' +
                                        'Inconsistency Rate: %{y:.1%}<br>' +
                                        'Inconsistent Cases: %{customdata[0]}<br>' +
                                        'Total Cases: %{customdata[1]}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="Inconsistency Rates by Category and Model",
                        xaxis_title="Category",
                        yaxis_title="Inconsistency Rate",
                        yaxis=dict(range=[0, 1], tickformat='.0%'),
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed category table
                    st.subheader("Detailed Category Inconsistency Metrics")
                    display_category_df = category_inconsistency_df.copy()
                    display_category_df['Inconsistency Rate'] = display_category_df['Inconsistency Rate'].apply(lambda x: f"{x:.1%}")
                    display_category_df['95% CI'] = display_category_df.apply(
                        lambda row: f"[{row['CI Lower']:.1%}, {row['CI Upper']:.1%}]", axis=1
                    )
                    display_category_df = display_category_df[['Category', 'Model', 'Inconsistency Rate', '95% CI', 'Inconsistent Cases', 'Total Cases']]
                    st.dataframe(display_category_df, use_container_width=True)

if __name__ == "__main__":
    main() 