import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import html
import json
from utils import (
    # Configuration and constants
    RESULTS_DIR, EXCLUDED_FILES, INCONSISTENCY_OPTIONS, 
    BOOLEAN_CONVERSION_OPTIONS, PERFORMANCE_CATEGORIES,
    # Page setup
    setup_page_config,
    # Data loading and processing
    load_jsonl, discover_jsonl_files, get_grader_info,
    # Data conversion
    convert_simulatability_to_bool, apply_boolean_conversion,
    # Calculation functions
    calculate_grader_performance, calculate_performance_metrics,
    # Helper functions
    get_available_graders
)

# Set page configuration
setup_page_config("Verbalization Evaluation Explorer")

def format_grader_name(grader_name):
    """Convert grader name to display format"""
    return grader_name.replace('faithfulness_', '').replace('_', ' ').title()

def get_applicable_graders(category, grader_info, df=None):
    """Get graders applicable to a specific category"""
    # Try to get graders from grader_info first
    if grader_info and 'category_graders' in grader_info:
        category_graders = grader_info['category_graders'].get(category, [])
        if category_graders:
            return category_graders
    
    # Fallback: detect graders from available columns in the dataframe
    if df is not None:
        available_graders = []
        grader_columns = [col for col in df.columns if col.endswith('_pred')]
        
        for col in grader_columns:
            # Extract grader name by removing '_pred' suffix
            grader_name = col.replace('_pred', '')
            available_graders.append(grader_name)
        
        return available_graders
    
    # Final fallback: return common graders if they exist in grader_info
    if grader_info and 'grader_names' in grader_info:
        return grader_info['grader_names']
    
    return []

def get_category_grader_pairs():
    """Get mapping of category to (simulatability_grader, tuned_simulatability_grader)"""
    # Now we just have two graders for each category: simulatability and tuned simulatability
    from graders import ALL_FAITHFULNESS_GRADER_PROMPTS
    
    pairs = {}
    category_specific_graders = ALL_FAITHFULNESS_GRADER_PROMPTS['faithfulness_simulability_tuned']
    
    for category in category_specific_graders.keys():
        pairs[category] = (
            'faithfulness_simulability',        # Simulatability grader (same for all)
            'faithfulness_simulability_tuned'    # Tuned simulatability grader
        )
    
    return pairs

def generate_record_id(record, file_stem):
    """Generate a unique, consistent record ID based on record content"""
    return f"{file_stem}_{record.get('category', 'unknown')}_{record.get('subcategory', 'unknown')}_{str(hash(str(record.get('prompt', ''))[:100]))}"

def display_record_metadata(record, show_manual=False, selected_file=None, record_idx=None, performance_info=""):
    """Display record metadata in a flexible column layout"""
    base_cols = ['category', 'subcategory', 'inconsistency_label']
    extra_cols = []
    
    if show_manual:
        extra_cols.append('manual_label')
    if performance_info:
        extra_cols.append('performance')
    if 'baseline_inconsistency_label' in record:
        extra_cols.append('baseline_inconsistency_label')
    
    cols = st.columns(len(base_cols) + len(extra_cols))
    
    # Standard columns
    with cols[0]:
        st.info(f"**Category:** {record['category']}")
    with cols[1]:
        st.info(f"**Subcategory:** {record['subcategory']}")
    with cols[2]:
        status = "Inconsistent" if record['inconsistency_label'] else "Consistent"
        st.info(f"**Inconsistency:** {status}")
    
    col_idx = 3
    
    # Extra columns
    if 'baseline_inconsistency_label' in record:
        with cols[col_idx]:
            status = "Resampled Inconsistent" if record.get('baseline_inconsistency_label', False) else "Resampled Consistent"
            st.info(f"**Resampled:** {status}")
        col_idx += 1
    
    if performance_info:
        with cols[col_idx]:
            st.info(performance_info)
        col_idx += 1
    
    if show_manual and selected_file and record_idx is not None:
        with cols[col_idx]:
            record_id = f"{selected_file.stem}_{record_idx}"
            manual_label = st.session_state.manual_labels.get(record_id, None)
            if manual_label is not None:
                st.info(f"**Manual:** {'✓ Faithful' if manual_label else '✗ Unfaithful'}")
            else:
                st.info("**Manual:** Not labeled")

def display_prompt_response_comparison(record, mode="original_vs_perturbed"):
    """Display prompt and response comparison based on mode"""
    col1, col2 = st.columns(2)
    
    if mode == "original_vs_perturbed":
        with col1:
            st.subheader("Original Prompt & Response")
            if 'prompt' in record and record['prompt']:
                prompt_text = record['prompt'][0]['content'] if isinstance(record['prompt'], list) else str(record['prompt'])
                display_formatted_text(prompt_text, "Original Prompt")
            if 'model_response' in record and record['model_response']:
                display_formatted_text(record['model_response'], "Original Response")
            if 'parsed_model_response' in record and record['parsed_model_response']:
                display_formatted_text(record['parsed_model_response'], "Parsed Original Response")
        
        with col2:
            st.subheader("Perturbed Prompt & Response")
            if 'perturbed_prompt' in record and record['perturbed_prompt']:
                perturbed_prompt_text = record['perturbed_prompt'][-1]['content'] if isinstance(record['perturbed_prompt'], list) else str(record['perturbed_prompt'])
                display_formatted_text(perturbed_prompt_text, "Perturbed Prompt")
            if 'perturbed_model_response' in record and record['perturbed_model_response']:
                display_formatted_text(record['perturbed_model_response'], "Perturbed Response")
            if 'parsed_perturbed_model_response' in record and record['parsed_perturbed_model_response']:
                display_formatted_text(record['parsed_perturbed_model_response'], "Parsed Perturbed Response")
    
    elif mode == "original_vs_resampled":
        with col1:
            st.subheader("Original Prompt & Response 1")
            if 'prompt' in record and record['prompt']:
                prompt_text = record['prompt'][0]['content'] if isinstance(record['prompt'], list) else str(record['prompt'])
                display_formatted_text(prompt_text, "Original Prompt")
            if 'model_response' in record and record['model_response']:
                display_formatted_text(record['model_response'], "Original Response 1")
            if 'parsed_model_response' in record and record['parsed_model_response']:
                display_formatted_text(record['parsed_model_response'], "Parsed Original Response 1")
        
        with col2:
            st.subheader("Original Prompt & Response 2")
            if 'prompt' in record and record['prompt']:
                prompt_text = record['prompt'][0]['content'] if isinstance(record['prompt'], list) else str(record['prompt'])
                display_formatted_text(prompt_text, "Original Prompt (Same)")
            if 'baseline_model_response' in record and record['baseline_model_response']:
                display_formatted_text(record['baseline_model_response'], "Original Response 2")
            if 'parsed_baseline_model_response' in record and record['parsed_baseline_model_response']:
                display_formatted_text(record['parsed_baseline_model_response'], "Parsed Original Response 2")

def auto_save_label(record_id):
    """Callback function to automatically save label when radio button changes"""
    # Get the new value from the radio button
    radio_key = f"manual_grade_{record_id}"
    if radio_key in st.session_state:
        selected_label = st.session_state[radio_key]
        new_label = True if selected_label == "True (Faithful)" else (False if selected_label == "False (Unfaithful)" else None)
        
        # Save to session state
        st.session_state.manual_labels[record_id] = new_label
        
        # Show feedback
        status_msg = "True" if new_label == True else "False" if new_label == False else "None"
        st.toast(f"✅ Auto-saved as {status_msg}", icon="💾")

def render_manual_grading_interface(record, record_id):
    """Compact manual grading interface with auto-save"""
    if not st.session_state.get('manual_grading_enabled', False):
        return
    
    st.markdown("---")
    st.subheader("Manual Faithfulness Grading")
    
    current_label = st.session_state.manual_labels.get(record_id, None)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        label_options = ["Skip (None)", "True (Faithful)", "False (Unfaithful)"]
        default_idx = 2 if current_label is None else (1 if current_label else 2)
        
        selected_label = st.radio(
            "Manual Faithfulness Label:", 
            label_options, 
            index=default_idx, 
            key=f"manual_grade_{record_id}",
            help="True = Response is faithful to the original context. Changes auto-save!",
            on_change=auto_save_label,
            args=(record_id,)
        )
        
        # Show current record ID for debugging
        st.caption(f"Record ID: {record_id}")
    
    with col2:
        if current_label is not None:
            status_color = "green" if current_label else "red"
            status_text = "Faithful" if current_label else "Unfaithful"
            st.markdown(f"**Current:** ::{status_color}[{status_text}]")
        
        # Show auto-save status
        st.info("🔄 Auto-saves on change")

def display_grader_analysis(record, grader_name, grader_true_values=None):
    """Display analysis for a specific grader"""
    display_name = format_grader_name(grader_name)
    st.subheader(f"{display_name} Analysis")
    
    response_col = f"{grader_name}_response"
    pred_col = f"{grader_name}_pred"
    
    # Display grader response if available
    if record.get(response_col):
        display_formatted_text(record[response_col], f"{display_name} Evaluation", max_height=400)
    
    # Display prediction
    pred_value = record.get(pred_col)
    if grader_name in ['faithfulness_simulability', 'faithfulness_simulability_tuned'] and grader_true_values is not None:
        pred_bool = convert_simulatability_to_bool(pred_value, grader_true_values)
    else:
        pred_bool = bool(pred_value)
    
    pred_status = "TRUE" if pred_bool else "FALSE"
    if grader_name in ['faithfulness_simulability', 'faithfulness_simulability_tuned']:
        st.markdown(f"**Raw Prediction:** {pred_value}")
    st.markdown(f"**Boolean Prediction:** {pred_status}")

def display_formatted_text(text, title="", max_height=200):
    """Display formatted text in an expandable container"""
    if text:
        with st.expander(title, expanded=True):
            # Use HTML escaping to preserve XML tags while keeping custom styling
            escaped_text = html.escape(text)
            st.markdown(f'<div style="max-height: {max_height}px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; font-size: 12px; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">{escaped_text}</div>', unsafe_allow_html=True)

def create_overview_metrics(filtered_df, grader_info, grader_true_values):
    """Create overview metrics section"""
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(filtered_df))
    
    with col2:
        inconsistent_count = len(filtered_df[filtered_df['inconsistency_label'] == True])
        st.metric("Inconsistent Records", inconsistent_count)
    
    with col3:
        inconsistency_rate = inconsistent_count / len(filtered_df) if len(filtered_df) > 0 else 0
        st.metric("Inconsistency Rate", f"{inconsistency_rate:.1%}")
    
    with col4:
        unique_categories = filtered_df['category'].nunique()
        st.metric("Categories", unique_categories)

def create_performance_metrics_section(filtered_df, grader_info, grader_true_values):
    """Create performance metrics section for graders"""
    
    if len(filtered_df) == 0:
        st.info("No data available for performance metrics.")
        return
    
    # Calculate performance for each grader
    performance_data = []
    
    for category in filtered_df['category'].unique():
        category_df = filtered_df[filtered_df['category'] == category]
        applicable_graders = get_applicable_graders(category, grader_info, category_df)
        
        for grader_name in applicable_graders:
            pred_col = f"{grader_name}_pred"
            
            if pred_col in category_df.columns:
                try:
                    performance = calculate_grader_performance(
                        category_df, grader_name, grader_true_values=grader_true_values
                    )
                    
                    performance_data.append({
                        'Category': category,
                        'Grader': format_grader_name(grader_name),
                        'Accuracy': performance.get('accuracy', 0),
                        'Precision': performance.get('precision', 0),
                        'Recall': performance.get('recall', 0),
                        'F1': performance.get('f1_score', 0)
                    })
                except Exception as e:
                    st.warning(f"Could not calculate performance for {grader_name} on {category}: {str(e)}")
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        

        
        # Calculate aggregate metrics
        st.subheader("Aggregate Performance Metrics")
        
        # Simple approach: calculate overall aggregate by pooling all data
        aggregate_metrics = []
        grader_name_map = {
            'Simulability Tuned': 'faithfulness_simulability_tuned',
            'Simulability': 'faithfulness_simulability'
        }
        
        for display_name, grader_name in grader_name_map.items():
            pred_col = f"{grader_name}_pred"
            if pred_col in filtered_df.columns:
                try:
                    # Get overall predictions across all categories
                    y_true = filtered_df['inconsistency_label'].fillna(False).astype(bool)
                    y_pred = apply_boolean_conversion(filtered_df, grader_name, grader_true_values)
                    
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true, y_pred)
                    
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        total_samples = len(filtered_df)
                        
                        # Calculate all metrics
                        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        aggregate_metrics.append({
                            'Grader': display_name,
                            'Total_Samples': total_samples,
                            'TP': int(tp),
                            'FP': int(fp),
                            'TN': int(tn),
                            'FN': int(fn),
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall (TPR)': recall,
                            'Specificity (TNR)': specificity,
                            'FPR': fpr,
                            'FNR': fnr,
                            'F1_Score': f1
                        })
                except Exception as e:
                    st.warning(f"Could not calculate aggregate metrics for {display_name}: {str(e)}")
        
        if aggregate_metrics:
            agg_df = pd.DataFrame(aggregate_metrics)
            
            # Display confusion matrix counts
            st.markdown("**Confusion Matrix Counts**")
            conf_cols = ['Grader', 'Total_Samples', 'TP', 'FP', 'TN', 'FN']
            st.dataframe(agg_df[conf_cols], use_container_width=True)
            
            # Display performance metrics in a visual format
            st.markdown("**Performance Metrics**")
            
            # Create columns for each grader
            grader_cols = st.columns(len(agg_df))
            
            for idx, (_, row) in enumerate(agg_df.iterrows()):
                with grader_cols[idx]:
                    st.markdown(f"### {row['Grader']}")
                    
                    # Primary metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{row['Accuracy']:.3f}")
                        st.metric("Precision", f"{row['Precision']:.3f}")
                        st.metric("F1 Score", f"{row['F1_Score']:.3f}")
                    
                    with col2:
                        st.metric("Recall (TPR)", f"{row['Recall (TPR)']:.3f}")
                        st.metric("Specificity (TNR)", f"{row['Specificity (TNR)']:.3f}")
                        st.metric("FPR", f"{row['FPR']:.3f}")
                    
                    # Additional metrics in smaller text
                    st.markdown(f"**FNR:** {row['FNR']:.3f}")
                    st.markdown("---")
            

        
        # Summary by category and grader (simplified)
        st.subheader("Category-wise Performance Summary")
        category_summary = perf_df.pivot(index='Category', columns='Grader', values=['Accuracy', 'Precision', 'Recall', 'F1'])
        category_summary = category_summary.round(3)
        st.dataframe(category_summary, use_container_width=True)
    else:
        st.info("No performance data available for the current filters.")

def create_navigation_controls(filtered_df, session_prefix, session_key, format_function, label=None):
    """Create navigation controls for browsing through records"""
    if len(filtered_df) == 0:
        return 0
    
    # Default label if none provided
    if label is None:
        label = "Select record:"
    
    # Create options for selectbox using the format function
    options = [format_function(i) for i in range(len(filtered_df))]
    
    # Use a separate key for internal navigation tracking
    nav_key = f"{session_key}_nav_idx"
    
    # Get current index from session state, defaulting to 0
    current_idx = st.session_state.get(nav_key, 0)
    
    # Ensure current_idx is within bounds
    if current_idx >= len(filtered_df):
        current_idx = 0
        st.session_state[nav_key] = current_idx
    
    # Handle navigation buttons
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("← Previous", key=f"{session_prefix}_prev"):
            new_idx = max(0, current_idx - 1)
            st.session_state[nav_key] = new_idx
            st.rerun()
    
    with col3:
        if st.button("Next →", key=f"{session_prefix}_next"):
            new_idx = min(len(filtered_df) - 1, current_idx + 1)
            st.session_state[nav_key] = new_idx
            st.rerun()
    
    with col2:
        # Use the navigation index for the selectbox
        selected_idx = st.selectbox(
            label,
            options=list(range(len(filtered_df))),
            format_func=lambda x: options[x],
            index=current_idx,
            key=session_key
        )
        
        # Update navigation tracking if user changed selection via selectbox
        if selected_idx != current_idx:
            st.session_state[nav_key] = selected_idx
    
    return selected_idx

def create_sidebar_filters(df):
    """Create sidebar filters and return selected values"""
    st.sidebar.header("Filters")
    
    # Category filter
    categories = ["All"] + sorted(df['category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Category:", categories)
    
    # Subcategory filter
    if selected_category != "All":
        subcategories = ["All"] + sorted(df[df['category'] == selected_category]['subcategory'].dropna().unique().tolist())
    else:
        subcategories = ["All"] + sorted(df['subcategory'].dropna().unique().tolist())
    selected_subcategory = st.sidebar.selectbox("Subcategory:", subcategories)
    
    # Inconsistency label filter
    selected_inconsistency = st.sidebar.selectbox("Inconsistency Label:", INCONSISTENCY_OPTIONS)
    
    # Boolean conversion settings
    st.sidebar.subheader("Faithfulness Boolean Conversion")
    selected_conversion = st.sidebar.selectbox(
        "What counts as True (unfaithful) for grader predictions?",
        options=list(BOOLEAN_CONVERSION_OPTIONS.keys()),
        index=0,
        help="Select which values should be considered True (unfaithful) for grader predictions"
    )
    grader_true_values = BOOLEAN_CONVERSION_OPTIONS[selected_conversion]
    
    return selected_category, selected_subcategory, selected_inconsistency, grader_true_values

def main():
    st.title("Verbalization Evaluation Explorer")
    st.markdown("---")
    
    # File selection
    st.sidebar.header("Data Selection")
    
    # Cache control
    if st.sidebar.button("🔄 Clear Cache & Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Look for JSONL files in the results directory
    jsonl_files = discover_jsonl_files()
    
    if not jsonl_files:
        st.error("No JSONL files found in the results directory.")
        st.info("Expected structure: results/{model_name}/{dataset_name}_{timestamp}_{config}.jsonl")
        return
    
    selected_file = st.sidebar.selectbox(
        "Select data file:",
        jsonl_files,
        format_func=lambda x: str(x.relative_to(RESULTS_DIR))
    )
    
    if selected_file:
        # Get file modification time to invalidate cache when file changes
        file_mtime = selected_file.stat().st_mtime
        df = load_jsonl(selected_file, _last_modified=file_mtime)
        
        if df is None:
            return
        
        st.sidebar.success(f"Loaded {len(df)} records from {selected_file.name}")
        
        # Sidebar filters
        selected_category, selected_subcategory, selected_inconsistency, grader_true_values = create_sidebar_filters(df)
        
        # Performance filter (TP/FP/TN/FN) - only show if we have grader data
        grader_info = get_grader_info()
        selected_performance, grader_name = ("All", None)
        
        if grader_info['grader_names']:
            performance_grader = st.sidebar.selectbox(
                "Select Grader for Performance Filter:",
                ["None"] + [format_grader_name(name) for name in grader_info['grader_names']],
                key="performance_grader_filter"
            )
            
            if performance_grader != "None":
                grader_name = next(name for name in grader_info['grader_names'] 
                                 if format_grader_name(name) == performance_grader)
                selected_performance = st.sidebar.selectbox("Performance Category:", PERFORMANCE_CATEGORIES)
        
        if selected_performance == "None":
            selected_performance = "All"
            grader_name = None
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply category filter
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        # Apply subcategory filter
        if selected_subcategory != "All":
            filtered_df = filtered_df[filtered_df['subcategory'] == selected_subcategory]
        
        # Apply inconsistency filter
        if selected_inconsistency != "All":
            if selected_inconsistency == "Inconsistent (True)":
                filtered_df = filtered_df[filtered_df['inconsistency_label'] == True]
            else:
                filtered_df = filtered_df[filtered_df['inconsistency_label'] == False]
        
        # Apply performance filter
        if selected_performance != "All" and grader_name:
            pred_col = f"{grader_name}_pred"
            if pred_col in filtered_df.columns:
                y_true = filtered_df['inconsistency_label'].fillna(False).astype(bool)
                y_pred = apply_boolean_conversion(filtered_df, grader_name, grader_true_values)
                
                if selected_performance == "True Positives":
                    mask = (y_true == True) & (y_pred == True)
                elif selected_performance == "False Positives":
                    mask = (y_true == False) & (y_pred == True)
                elif selected_performance == "True Negatives":
                    mask = (y_true == False) & (y_pred == False)
                elif selected_performance == "False Negatives":
                    mask = (y_true == True) & (y_pred == False)
                
                filtered_df = filtered_df[mask]
        
        st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)} records")
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Inconsistency Analysis", "resample Inconsistency Analysis", "Verbalization Analysis"])
        
        with tab1:
            create_overview_metrics(filtered_df, grader_info, grader_true_values)
            
            # Verbalization performance metrics section
            create_performance_metrics_section(filtered_df, grader_info, grader_true_values)
            
            # Visualizations
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Categories Distribution")
                category_counts = filtered_df['category'].value_counts()
                fig = px.pie(values=category_counts.values, names=category_counts.index, 
                            title="Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Inconsistency by Category")
                category_inconsistency = filtered_df.groupby('category')['inconsistency_label'].agg(['count', 'sum']).reset_index()
                category_inconsistency['percentage'] = (category_inconsistency['sum'] / category_inconsistency['count']) * 100
                
                # Calculate resampled inconsistency
                resampled_inconsistency = filtered_df.groupby('category')['baseline_inconsistency_label'].agg(['count', 'sum']).reset_index()
                resampled_inconsistency['resampled_percentage'] = (resampled_inconsistency['sum'] / resampled_inconsistency['count']) * 100
                
                # Merge the data
                merged_data = pd.merge(category_inconsistency, resampled_inconsistency[['category', 'resampled_percentage']], on='category', how='left')
                
                # Create the bar chart with grouped bars
                fig = go.Figure()
                
                # Add main inconsistency bars
                fig.add_trace(go.Bar(
                    x=merged_data['category'],
                    y=merged_data['percentage'],
                    name='Perturbed Inconsistency Rate',
                    marker_color='lightcoral',
                    opacity=0.8
                ))
                
                # Add resampled inconsistency as grouped bars
                fig.add_trace(go.Bar(
                    x=merged_data['category'],
                    y=merged_data['resampled_percentage'],
                    name='Resampled Inconsistency Rate',
                    marker_color='lightblue',
                    opacity=0.8
                ))
                
                fig.update_layout(
                    title="Inconsistency Rate by Category (Perturbed vs Resampled)",
                    xaxis_title="Category",
                    yaxis_title="Inconsistency Rate (%)",
                    barmode='group',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Inconsistency Analysis")
            
            if len(filtered_df) > 0:
                # Manual grading controls
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if st.button("Enable Manual Grading", key="enable_manual_grading"):
                        # Initialize manual grading session state
                        if 'manual_grading_enabled' not in st.session_state:
                            st.session_state.manual_grading_enabled = True
                        
                        # Add manual_faithfulness_label column if it doesn't exist
                        if 'manual_labels' not in st.session_state:
                            st.session_state.manual_labels = {}
                        
                        # Initialize manual labels for all records in current dataset
                        for idx in range(len(df)):  # Use original df, not filtered
                            record = df.iloc[idx]
                            record_id = generate_record_id(record, selected_file.stem)
                            if record_id not in st.session_state.manual_labels:
                                st.session_state.manual_labels[record_id] = None
                        
                        st.success("Manual grading enabled! Navigate through records to add labels.")
                
                with col2:
                    if st.session_state.get('manual_grading_enabled', False):
                        st.info("🔄 Auto-saving enabled")
                
                with col3:
                    if st.session_state.get('manual_grading_enabled', False):
                        manual_count = sum(1 for label in st.session_state.manual_labels.values() if label is not None)
                        st.metric("Labeled", f"{manual_count}/{len(df)}")
                        
                        # Debug button to show current labels
                        if st.button("🔍 Debug Labels", key="debug_labels"):
                            st.write("Current manual labels:")
                            for key, value in st.session_state.manual_labels.items():
                                if value is not None:
                                    st.write(f"- {key}: {value}")
                
                st.markdown("---")
                # Navigation and record display
                def format_inconsistency_record(x):
                    record = filtered_df.iloc[x]
                    status = "Inconsistent" if record['inconsistency_label'] else "Consistent"
                    if st.session_state.get('manual_grading_enabled', False):
                        record_id = generate_record_id(record, selected_file.stem)
                        manual_label = st.session_state.manual_labels.get(record_id, None)
                        if manual_label is not None:
                            manual_str = f"Manual: {'✓' if manual_label else '✗'}"
                            return f"Record {x + 1} - {record['category']} - {status} - {manual_str}"
                    return f"Record {x + 1} - {record['category']} - {status}"
                
                record_idx = create_navigation_controls(filtered_df, "inconsistency", "inconsistency_selector", format_inconsistency_record)
                record = filtered_df.iloc[record_idx]
                
                # Generate a unique record ID based on content
                record_id = generate_record_id(record, selected_file.stem)
                
                # Display metadata and manual grading
                display_record_metadata(record, show_manual=st.session_state.get('manual_grading_enabled', False), 
                                      selected_file=selected_file, record_idx=record_idx)
                
                render_manual_grading_interface(record, record_id)
                
                # Display prompts and responses
                display_prompt_response_comparison(record, mode="original_vs_perturbed")
                
                # Display chain of thought for perturbed response
                if 'perturbed_chain_of_thought' in record and record['perturbed_chain_of_thought']:
                    st.subheader("Chain of Thought (Perturbed)")
                    display_formatted_text(record['perturbed_chain_of_thought'], "Chain of Thought Reasoning", max_height=300)
                
                # Display ground truth if available
                if 'ground_truth' in record and record['ground_truth']:
                    st.subheader("Ground Truth")
                    st.success(f"**Ground Truth:** {record['ground_truth']}")
            else:
                st.info("No records match the current filters.")
        
        with tab3:
            st.header("Resampled Inconsistency Analysis")
            
            if len(filtered_df) > 0:
                # Navigation and record display
                def format_resampled_record(x):
                    record = filtered_df.iloc[x]
                    status = "Resampled Inconsistent" if record.get('baseline_inconsistency_label', False) else "Resampled Consistent"
                    return f"Record {x + 1} - {record['category']} - {status}"
                
                record_idx = create_navigation_controls(filtered_df, "resampled_inconsistency", "resampled_inconsistency_selector", format_resampled_record)
                record = filtered_df.iloc[record_idx]
                
                # Display metadata (including resampled status)
                display_record_metadata(record)
                
                # Display prompts and responses
                display_prompt_response_comparison(record, mode="original_vs_resampled")
                
                # Display comparison information
                st.subheader("Resampled Inconsistency Analysis")
                st.info("This analysis compares two responses to the same original prompt to measure natural model inconsistency due to sampling temperature.")
                
                # Show the comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Response 1 (Parsed):**")
                    st.code(record.get('parsed_model_response', 'N/A'))
                with col2:
                    st.markdown("**Response 2 (Parsed):**")
                    st.code(record.get('parsed_baseline_model_response', 'N/A'))
                
                # Display ground truth if available
                if 'ground_truth' in record and record['ground_truth']:
                    st.subheader("Ground Truth")
                    st.success(f"**Ground Truth:** {record['ground_truth']}")
            else:
                st.info("No records match the current filters.")
        
        with tab4:
            st.header("Verbalization Analysis")
            
            if len(filtered_df) > 0:
                # Navigation and record display
                grader_info = get_grader_info()
                
                def format_verbalization_record(x):
                    record = filtered_df.iloc[x]
                    applicable_graders = get_applicable_graders(record['category'], grader_info, filtered_df)
                    grader_parts = []
                    
                    for grader_name in applicable_graders:
                        pred_col = f"{grader_name}_pred"
                        pred_value = record.get(pred_col, 'N/A')
                        short_name = format_grader_name(grader_name)[:3]
                        grader_parts.append(f"{short_name}: {pred_value}")
                    
                    grader_str = " - ".join(grader_parts) if grader_parts else "No graders"
                    return f"Record {x + 1} - {record['category']} - {grader_str}"
                
                record_idx = create_navigation_controls(filtered_df, "verbalization", "verbalization_selector", 
                                                      format_verbalization_record, "Select record for verbalization analysis:")
                record = filtered_df.iloc[record_idx]
                
                # Display metadata with grader predictions
                applicable_graders = get_applicable_graders(record['category'], grader_info, filtered_df)
                
                # Performance info if filtering is active
                performance_info = ""
                if selected_performance != "All" and 'grader_name' in locals():
                    pred_col = f"{grader_name}_pred"
                    if pred_col in record.index:
                        y_true = bool(record.get('inconsistency_label', False))
                        # Create single-row dataframe for conversion
                        record_df = pd.DataFrame([record])
                        y_pred = apply_boolean_conversion(record_df, grader_name, grader_true_values).iloc[0]
                        
                        if y_true and y_pred:
                            performance_info = f"**{selected_performance}** (True Positive)"
                        elif not y_true and y_pred:
                            performance_info = f"**{selected_performance}** (False Positive)"
                        elif not y_true and not y_pred:
                            performance_info = f"**{selected_performance}** (True Negative)"
                        elif y_true and not y_pred:
                            performance_info = f"**{selected_performance}** (False Negative)"
                
                # Display extended metadata with grader predictions and manual label
                base_cols = 2  # category + inconsistency
                extra_cols = 1 if performance_info else 0
                manual_col = 1 if st.session_state.get('manual_grading_enabled', False) else 0
                num_meta_cols = base_cols + extra_cols + manual_col + len(applicable_graders)
                meta_cols = st.columns(num_meta_cols)
                
                with meta_cols[0]:
                    st.info(f"**Category:** {record['category']}")
                with meta_cols[1]:
                    inconsistency_status = "Inconsistent" if record['inconsistency_label'] else "Consistent"
                    st.info(f"**Inconsistency:** {inconsistency_status}")
                
                col_idx = 2
                if performance_info:
                    with meta_cols[col_idx]:
                        st.info(performance_info)
                    col_idx += 1
                
                # Display manual label if available
                if st.session_state.get('manual_grading_enabled', False):
                    with meta_cols[col_idx]:
                        record_id = generate_record_id(record, selected_file.stem)
                        manual_label = st.session_state.manual_labels.get(record_id, None)
                        if manual_label is not None:
                            status = "✓ Faithful" if manual_label else "✗ Unfaithful"
                            st.info(f"**Manual:** {status}")
                        else:
                            # Check if manual label exists in the data itself
                            if 'manual_faithfulness_label' in record and record['manual_faithfulness_label'] is not None:
                                status = "✓ Faithful" if record['manual_faithfulness_label'] else "✗ Unfaithful"
                                st.info(f"**Manual:** {status}")
                            else:
                                # Default to False (unfaithful) instead of "Not labeled"
                                st.info("**Manual:** ✗ Unfaithful")
                    col_idx += 1
                
                # Display grader predictions
                for i, grader_name in enumerate(applicable_graders):
                    with meta_cols[col_idx + i]:
                        pred_col = f"{grader_name}_pred"
                        status = "True" if record.get(pred_col) else "False"
                        display_name = format_grader_name(grader_name)
                        st.info(f"**{display_name}:** {status}")
                
                # Verbalization evaluations - dynamic display
                if applicable_graders:
                    num_graders = len(applicable_graders)
                    if num_graders == 1:
                        for grader_name in applicable_graders:
                            display_grader_analysis(record, grader_name, grader_true_values)
                    elif num_graders == 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            display_grader_analysis(record, applicable_graders[0], grader_true_values)
                        with col2:
                            display_grader_analysis(record, applicable_graders[1], grader_true_values)
                    else:
                        col1, col2 = st.columns(2)
                        for i, grader_name in enumerate(applicable_graders):
                            with col1 if i % 2 == 0 else col2:
                                display_grader_analysis(record, grader_name, grader_true_values)
                else:
                    st.info("No verbalization graders applicable to this category.")
                
                # Display context
                st.subheader("Context: Perturbed Response")
                if 'parsed_perturbed_model_response' in record and record['parsed_perturbed_model_response']:
                    display_formatted_text(record['parsed_perturbed_model_response'], "Perturbed Model Response", max_height=200)
                
                if 'perturbed_chain_of_thought' in record and record['perturbed_chain_of_thought']:
                    st.subheader("Context: Chain of Thought")
                    display_formatted_text(record['perturbed_chain_of_thought'], "Chain of Thought Reasoning", max_height=250)
            else:
                st.info("No records match the current filters.")
        


if __name__ == "__main__":
    main() 