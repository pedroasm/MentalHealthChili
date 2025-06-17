import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Mental Health Data Validation 2018", 
    page_icon="🧠", 
    layout="wide"
)

def load_default_data():
    """Load default data - exact from provided CSV"""
    data = {
        "region": [
            "Arica y Parinacota", "Tarapaca", "Antofagasta", "Atacama", "Coquimbo",
            "Valparaiso", "RM", "O'Higgins", "Maule",
            "Biobio", "Araucania", "Los Rios", "Los Lagos", "Aysen", "Magallanes"
        ],
        "population": [
            253620, 391084, 692557, 314569, 811812, 1985196, 8124762, 993826,
            1050522, 1616617, 1036290, 405835, 872498, 107297, 166533
        ],
        "general_care": [
            40119, 74010, 88337, 40187, 139327, 359035, 1317245, 197137, 221183,
            441948, 158772, 74177, 158135, 24987, 49264
        ],
        "medical_consultations": [
            11312, 21671, 22838, 11648, 28807, 80374, 284169, 49993, 77173,
            111720, 29291, 16315, 37278, 5599, 7126
        ],
        "psychological_care": [
            21012, 9795, 7052, 10684, 15587, 33035, 255880, 23023, 26351,
            71460, 60978, 13467, 26499, 11985, 10413
        ],
        "depression_care": [
            1056, 4459, 4035, 2388, 7983, 19193, 78843, 12291, 15603,
            19666, 12899, 5533, 11139, 1044, 2284
        ],
        "general_care_ratio": [
            6.32, 5.28, 7.84, 7.83, 5.83, 5.53, 6.17, 5.04, 4.75,
            3.66, 6.53, 5.47, 5.52, 4.29, 3.38
        ],
        "medical_consultations_ratio": [
            22.42, 18.05, 30.32, 27.01, 28.18, 24.7, 28.59, 19.88, 13.61,
            14.47, 35.38, 24.87, 23.41, 19.16, 23.37
        ],
        "psychological_care_ratio": [
            12.07, 39.93, 98.21, 29.44, 52.08, 60.09, 31.75, 43.17, 39.87,
            22.62, 16.99, 30.14, 32.93, 8.95, 15.99
        ],
        "depression_care_ratio": [
            240.17, 87.71, 171.64, 131.73, 101.69, 103.43, 103.05, 80.86, 67.33,
            82.20, 80.34, 73.35, 78.33, 102.77, 72.91
        ]
    }
    return pd.DataFrame(data)

def plot_2018_data(df):
    """Plot 2018 study data"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.bar(df['region'], df['population'], color='lightblue', alpha=0.8, edgecolor='navy')
    ax.set_title('2018 Study Data: Population by Region', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Population (Inhabitants)', fontsize=12)
    ax.set_xlabel('Regions', fontsize=12)
    ax.tick_params(axis='x', rotation=90, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits to give more space for labels
    max_pop = max(df['population'])
    ax.set_ylim(0, max_pop * 1.15)  # Add 15% more space at the top
    
    # Add values on bars
    for i, v in enumerate(df['population']):
        ax.text(i, v + max_pop * 0.02, f'{v:,.0f}', 
                ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    return fig

def plot_normalized_data(df):
    """Line plots for normalized data - original order"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Normalized Data: 2018 Population ÷ REM-9 Care (Original Order)\n(Lower = Better Coverage)', 
                 fontsize=16, fontweight='bold')
    
    # ROW 1: LINE PLOTS
    # Plot 1: Normalized General Care - Line
    axes[0,0].plot(df['region'], df['general_care_ratio'], 'o-', linewidth=3, markersize=8, color='green', markerfacecolor='lightgreen')
    axes[0,0].set_title('General Consultation')
    axes[0,0].set_ylabel('Consultation/Ratio')
    axes[0,0].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add average line
    avg_general = df['general_care_ratio'].mean()
    axes[0,0].axhline(y=avg_general, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0,0].text(0.02, 0.95, f'Average: {avg_general:.2f}', transform=axes[0,0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Plot 2: Normalized Medical Consultations - Line
    axes[0,1].plot(df['region'], df['medical_consultations_ratio'], 'o-', linewidth=3, markersize=8, color='blue', markerfacecolor='lightblue')
    axes[0,1].set_title('Medical Consultation')
    axes[0,1].set_ylabel('Consultation/Ratio')
    axes[0,1].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add average line
    avg_medical = df['medical_consultations_ratio'].mean()
    axes[0,1].axhline(y=avg_medical, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0,1].text(0.02, 0.95, f'Average: {avg_medical:.2f}', transform=axes[0,1].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Plot 3: Normalized Psychological Care - Line
    axes[0,2].plot(df['region'], df['psychological_care_ratio'], 'o-', linewidth=3, markersize=8, color='purple', markerfacecolor='plum')
    axes[0,2].set_title('Psychological Consultation')
    axes[0,2].set_ylabel('Consultation/Ratio')
    axes[0,2].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,2].grid(True, alpha=0.3)
    
    # Add average line
    avg_psych = df['psychological_care_ratio'].mean()
    axes[0,2].axhline(y=avg_psych, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0,2].text(0.02, 0.95, f'Average: {avg_psych:.2f}', transform=axes[0,2].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # ROW 2: BAR PLOTS
    # Plot 4: Normalized General Care - Bar
    axes[1,0].bar(df['region'], df['general_care_ratio'], color='lightgreen', alpha=0.8, edgecolor='green')
    axes[1,0].set_title('General Consultation (Bars)')
    axes[1,0].set_ylabel('Consultation/Ratio')
    axes[1,0].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].axhline(y=avg_general, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Plot 5: Normalized Medical Consultations - Bar
    axes[1,1].bar(df['region'], df['medical_consultations_ratio'], color='lightblue', alpha=0.8, edgecolor='blue')
    axes[1,1].set_title('Medical Consultation (Bars)')
    axes[1,1].set_ylabel('Consultation/Ratio')
    axes[1,1].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,1].grid(True, alpha=0.3, axis='y')
    axes[1,1].axhline(y=avg_medical, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Plot 6: Normalized Psychological Care - Bar
    axes[1,2].bar(df['region'], df['psychological_care_ratio'], color='plum', alpha=0.8, edgecolor='purple')
    axes[1,2].set_title('Psychological Consultation (Bars)')
    axes[1,2].set_ylabel('Consultation/Ratio')
    axes[1,2].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,2].grid(True, alpha=0.3, axis='y')
    axes[1,2].axhline(y=avg_psych, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    plt.tight_layout()
    return fig

def create_summary_stats(df):
    """Create summary statistics table for normalized data only (without depression)"""
    stats_data = {
        'Normalized Metric': ['General Consultation', 'Medical Consultation', 'Psychological Consultation'],
        'Average': [
            df['general_care_ratio'].mean(),
            df['medical_consultations_ratio'].mean(),
            df['psychological_care_ratio'].mean()
        ],
        'Best Coverage (Min)': [
            df['general_care_ratio'].min(),
            df['medical_consultations_ratio'].min(),
            df['psychological_care_ratio'].min()
        ],
        'Worst Coverage (Max)': [
            df['general_care_ratio'].max(),
            df['medical_consultations_ratio'].max(),     
            df['psychological_care_ratio'].max()
        ],
        'Best Coverage Region': [
            df.loc[df['general_care_ratio'].idxmin(), 'region'],
            df.loc[df['medical_consultations_ratio'].idxmin(), 'region'],
            df.loc[df['psychological_care_ratio'].idxmin(), 'region']
        ],
        'Worst Coverage Region': [
            df.loc[df['general_care_ratio'].idxmax(), 'region'],
            df.loc[df['medical_consultations_ratio'].idxmax(), 'region'],
            df.loc[df['psychological_care_ratio'].idxmax(), 'region']
        ]
    }
    return pd.DataFrame(stats_data)

def plot_normalized_data_by_population(df):
    """Line plots for normalized data - ordered by population (highest to lowest)"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Normalized Data: 2018 Population ÷ REM-9 Care (Ordered by Population)\n(Lower = Better Coverage)', 
                 fontsize=16, fontweight='bold')
    
    # Sort data by population (descending) - RM will be first, smallest population last
    df_sorted = df.sort_values('population', ascending=False).reset_index(drop=True)
    
    # ROW 1: LINE PLOTS
    # Plot 1: Normalized General Care - Line ordered by population
    axes[0,0].plot(df_sorted['region'], df_sorted['general_care_ratio'], 'o-', linewidth=3, markersize=8, color='green', markerfacecolor='lightgreen')
    axes[0,0].set_title('General Consultation\nOrdered by Population')
    axes[0,0].set_ylabel('Consultation/Ratio')
    axes[0,0].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add average line
    avg_general = df['general_care_ratio'].mean()
    axes[0,0].axhline(y=avg_general, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0,0].text(0.02, 0.95, f'Average: {avg_general:.2f}', transform=axes[0,0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Plot 2: Normalized Medical Consultations - Line ordered by population
    axes[0,1].plot(df_sorted['region'], df_sorted['medical_consultations_ratio'], 'o-', linewidth=3, markersize=8, color='blue', markerfacecolor='lightblue')
    axes[0,1].set_title('Medical Consultation\nOrdered by Population')
    axes[0,1].set_ylabel('Consultation/Ratio')
    axes[0,1].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add average line
    avg_medical = df['medical_consultations_ratio'].mean()
    axes[0,1].axhline(y=avg_medical, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0,1].text(0.02, 0.95, f'Average: {avg_medical:.2f}', transform=axes[0,1].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Plot 3: Normalized Psychological Care - Line ordered by population
    axes[0,2].plot(df_sorted['region'], df_sorted['psychological_care_ratio'], 'o-', linewidth=3, markersize=8, color='purple', markerfacecolor='plum')
    axes[0,2].set_title('Psychological Consultation\nOrdered by Population')
    axes[0,2].set_ylabel('Consultation/Ratio')
    axes[0,2].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,2].grid(True, alpha=0.3)
    
    # Add average line
    avg_psych = df['psychological_care_ratio'].mean()
    axes[0,2].axhline(y=avg_psych, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0,2].text(0.02, 0.95, f'Average: {avg_psych:.2f}', transform=axes[0,2].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # ROW 2: BAR PLOTS
    # Plot 4: Normalized General Care - Bar ordered by population
    axes[1,0].bar(df_sorted['region'], df_sorted['general_care_ratio'], color='lightgreen', alpha=0.8, edgecolor='green')
    axes[1,0].set_title('General Consultation (Bars)\nOrdered by Population')
    axes[1,0].set_ylabel('Consultation/Ratio')
    axes[1,0].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].axhline(y=avg_general, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Plot 5: Normalized Medical Consultations - Bar ordered by population
    axes[1,1].bar(df_sorted['region'], df_sorted['medical_consultations_ratio'], color='lightblue', alpha=0.8, edgecolor='blue')
    axes[1,1].set_title('Medical Consultation (Bars)\nOrdered by Population')
    axes[1,1].set_ylabel('Consultation/Ratio')
    axes[1,1].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,1].grid(True, alpha=0.3, axis='y')
    axes[1,1].axhline(y=avg_medical, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Plot 6: Normalized Psychological Care - Bar ordered by population
    axes[1,2].bar(df_sorted['region'], df_sorted['psychological_care_ratio'], color='plum', alpha=0.8, edgecolor='purple')
    axes[1,2].set_title('Psychological Consultation (Bars)\nOrdered by Population')
    axes[1,2].set_ylabel('Consultation/Ratio')
    axes[1,2].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,2].grid(True, alpha=0.3, axis='y')
    axes[1,2].axhline(y=avg_psych, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    plt.tight_layout()
    return fig

def plot_depression_data(df):
    """Line and bar plots for depression data - original order"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Depression Consultation Analysis\n(Lower = Better Coverage)', 
                 fontsize=16, fontweight='bold')
    
    # Calculate average
    avg_depression = df['depression_care_ratio'].mean()
    
    # Plot 1: Depression Care - Line (Original Order)
    axes[0,0].plot(df['region'], df['depression_care_ratio'], 'o-', linewidth=3, markersize=8, 
                   color='darkorange', markerfacecolor='orange')
    axes[0,0].set_title('Depression Consultation - Original Order', fontweight='bold')
    axes[0,0].set_ylabel('Consultation/Ratio')
    axes[0,0].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=avg_depression, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0,0].text(0.02, 0.95, f'Average: {avg_depression:.2f}', transform=axes[0,0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Plot 2: Depression Care - Bar (Original Order)
    axes[0,1].bar(df['region'], df['depression_care_ratio'], color='orange', alpha=0.8, edgecolor='darkorange')
    axes[0,1].set_title('Depression Consultation - Original Order (Bars)', fontweight='bold')
    axes[0,1].set_ylabel('Consultation/Ratio')
    axes[0,1].tick_params(axis='x', rotation=90, labelsize=9)
    axes[0,1].grid(True, alpha=0.3, axis='y')
    axes[0,1].axhline(y=avg_depression, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Sort data by population (descending)
    df_sorted = df.sort_values('population', ascending=False).reset_index(drop=True)
    
    # Plot 3: Depression Care - Line (Ordered by Population)
    axes[1,0].plot(df_sorted['region'], df_sorted['depression_care_ratio'], 'o-', linewidth=3, markersize=8, 
                   color='darkorange', markerfacecolor='orange')
    axes[1,0].set_title('Depression Consultation - Ordered by Population', fontweight='bold')
    axes[1,0].set_ylabel('Consultation/Ratio')
    axes[1,0].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=avg_depression, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[1,0].text(0.02, 0.95, f'Average: {avg_depression:.2f}', transform=axes[1,0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Plot 4: Depression Care - Bar (Ordered by Population)
    axes[1,1].bar(df_sorted['region'], df_sorted['depression_care_ratio'], color='orange', alpha=0.8, edgecolor='darkorange')
    axes[1,1].set_title('Depression Consultation - Ordered by Population (Bars)', fontweight='bold')
    axes[1,1].set_ylabel('Consultation/Ratio')
    axes[1,1].tick_params(axis='x', rotation=90, labelsize=9)
    axes[1,1].grid(True, alpha=0.3, axis='y')
    axes[1,1].axhline(y=avg_depression, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    plt.tight_layout()
    return fig
    """Create summary statistics table for normalized data only"""
    stats_data = {
        'Normalized Metric': ['General Consultation', 'Medical Consultation', 'Psychological Consultation', 'Depression Consultation'],
        'Average': [
            df['general_care_ratio'].mean(),
            df['medical_consultations_ratio'].mean(),
            df['psychological_care_ratio'].mean(),
            df['depression_care_ratio'].mean()
        ],
        'Best Coverage (Min)': [
            df['general_care_ratio'].min(),
            df['medical_consultations_ratio'].min(),
            df['psychological_care_ratio'].min(),
            df['depression_care_ratio'].min()
        ],
        'Worst Coverage (Max)': [
            df['general_care_ratio'].max(),
            df['medical_consultations_ratio'].max(),     
            df['psychological_care_ratio'].max(),
            df['depression_care_ratio'].max()
        ],
        'Best Coverage Region': [
            df.loc[df['general_care_ratio'].idxmin(), 'region'],
            df.loc[df['medical_consultations_ratio'].idxmin(), 'region'],
            df.loc[df['psychological_care_ratio'].idxmin(), 'region'],
            df.loc[df['depression_care_ratio'].idxmin(), 'region']
        ],
        'Worst Coverage Region': [
            df.loc[df['general_care_ratio'].idxmax(), 'region'],
            df.loc[df['medical_consultations_ratio'].idxmax(), 'region'],
            df.loc[df['psychological_care_ratio'].idxmax(), 'region'],
            df.loc[df['depression_care_ratio'].idxmax(), 'region']
        ]
    }
    return pd.DataFrame(stats_data)

def main():
    st.title("🧠 Mental Health Data Validation 2018")
    st.markdown("### Normalization with REM-9 Dataset for Cross Validation")
    
    # Methodological explanation
    with st.expander("🔬 Methodology Explanation"):
        st.markdown("""
        **Objective**: Validate 2018 study data using REM-9 as reference
        
        **Process**:
        1. **2018 Data**: Population by region (original study data)
        2. **REM-9 Data**: Actual care by region (official validation data)  
        3. **Normalization**: Population ÷ Care = Coverage ratio
        
        **Interpretation**: 
        - Low ratio = Good coverage (few people per care)
        - High ratio = Poor coverage (many people per care)
        """)
    
    # Sidebar options
    st.sidebar.title("Visualization Controls")
    
    # Option to load file or use default data
    use_default = st.sidebar.checkbox("Use default data", value=True)
    
    if use_default:
        df = load_default_data()
        st.success("✅ Data loaded successfully")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
                st.success("✅ File loaded successfully")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        else:
            st.warning("⚠️ Please upload a CSV file or check 'Use default data'")
            return
    
    # Show raw data table if requested
    if st.sidebar.checkbox("Show data table"):
        st.subheader("📋 Complete Data")
        st.dataframe(df, use_container_width=True)
    
    # Calculation validation
    if st.sidebar.checkbox("Validate calculations"):
        st.subheader("🔍 Normalized Calculations Validation")
        validation_df = df[['region', 'population', 'general_care', 'medical_consultations', 'psychological_care', 'depression_care']].copy()
        validation_df['general_care_calc'] = (validation_df['population'] / validation_df['general_care']).round(2)
        validation_df['medical_consultations_calc'] = (validation_df['population'] / validation_df['medical_consultations']).round(2)
        validation_df['psychological_care_calc'] = (validation_df['population'] / validation_df['psychological_care']).round(2)
        validation_df['depression_care_calc'] = (validation_df['population'] / validation_df['depression_care']).round(2)
        
        # Compare with original values
        validation_df['check_general'] = validation_df['general_care_calc'] == df['general_care_ratio']
        validation_df['check_medical'] = validation_df['medical_consultations_calc'] == df['medical_consultations_ratio']
        validation_df['check_psychological'] = validation_df['psychological_care_calc'] == df['psychological_care_ratio']
        validation_df['check_depression'] = validation_df['depression_care_calc'] == df['depression_care_ratio']
        
        st.dataframe(validation_df, use_container_width=True)
        
        # Validation summary
        all_checks = (validation_df['check_general'].all() and 
                     validation_df['check_medical'].all() and 
                     validation_df['check_psychological'].all() and
                     validation_df['check_depression'].all())
        
        if all_checks:
            st.success("✅ All calculations are correct")
        else:
            st.error("❌ There are discrepancies in calculations")
    
    # SECTION 1: 2018 DATA
    st.header("1️⃣ 2018 Study Data")
    st.markdown("*Population by region according to original risk factors study*")
    
    if st.sidebar.checkbox("Show 2018 data", value=True):
        fig_2018 = plot_2018_data(df)
        st.pyplot(fig_2018)
        
        # Basic 2018 statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Population", f"{df['population'].sum():,.0f}")
        with col2:
            st.metric("Most Populated Region", df.loc[df['population'].idxmax(), 'region'])
        with col3:
            st.metric("Least Populated Region", df.loc[df['population'].idxmin(), 'region'])
    
    st.divider()
    
    # SECTION 2: NORMALIZED DATA
    st.header("2️⃣ Normalized Data (Results)")
    st.markdown("*Calculated ratios: 2018 Population ÷ REM-9 Care*")
    st.markdown("")  # Add extra space
    st.markdown("")  # Add extra space
    
    if st.sidebar.checkbox("Show normalized data", value=True):
        # Summary statistics table
        st.subheader("📊 Statistical Summary")
        stats_df = create_summary_stats(df)
        st.dataframe(stats_df.round(2), use_container_width=True)
        
        # Normalized data visualization - Original order
        st.markdown("**Evolution of ratios by region (Original Order)**")
        fig_normalized = plot_normalized_data(df)
        st.pyplot(fig_normalized)
        
        st.divider()
        
        # Normalized data visualization - Ordered by population
        st.markdown("**Evolution of ratios by region (Ordered by Population)**")
        fig_normalized_population = plot_normalized_data_by_population(df)
        st.pyplot(fig_normalized_population)

    st.divider()
    
    # SECTION 3: DEPRESSION DATA SPECIFIC
    st.header("3️⃣ Depression Consultation Analysis")
    
    if st.sidebar.checkbox("Show depression analysis", value=True):
        fig_depression = plot_depression_data(df)
        st.pyplot(fig_depression)

if __name__ == "__main__":
    main()