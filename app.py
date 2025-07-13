import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import streamlit as st
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Frank-Starling Curve Analysis",
    page_icon="❤️",
    layout="wide"
)

# Simplified CSS styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card { 
        background-color: #fff; 
        padding: 1rem; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def logistic_function(x, A1, A2, x0, p):
    """4-parameter logistic function for Frank-Starling relationship"""
    return A2 + (A1 - A2) / (1 + (x/x0)**p)

def create_scatter_plot(x_data, y_data):
    """Create simple scatter plot when insufficient data for curve fitting"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(x_data, y_data, color='#2c3e50', s=80, label='Data Points', zorder=5)
    
    ax.set_xlabel("IV Volume (mL)", fontsize=12)
    ax.set_ylabel("Δ Velocity Time Integral (cm)", fontsize=12)
    ax.set_title("Frank-Starling Data (Scatter Plot)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig
def create_frank_starling_plot(x_data, y_data, fitted_params):
    """Create Frank-Starling curve plot with fitted parameters"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate smooth curve
    x_smooth = np.linspace(min(x_data)*0.8, max(x_data)*1.2, 200)
    y_smooth = logistic_function(x_smooth, *fitted_params)
    
    # Plot data and fitted curve
    ax.scatter(x_data, y_data, color='#2c3e50', s=80, label='Data Points', zorder=5)
    ax.plot(x_smooth, y_smooth, color='#e74c3c', linewidth=2.5, label='Frank-Starling Curve')
    
    # Mark key points
    ax.axvline(fitted_params[2], color='#3498db', linestyle='--', alpha=0.7, 
              label=f'Optimal Preload: {fitted_params[2]:.1f} mL')
    
    # Labels and formatting
    ax.set_xlabel("IV Volume (mL)", fontsize=12)
    ax.set_ylabel("Δ Velocity Time Integral (cm)", fontsize=12)
    ax.set_title("Frank-Starling Relationship", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add equation
    equation = f"A₁={fitted_params[0]:.2f}, A₂={fitted_params[1]:.2f}\nx₀={fitted_params[2]:.1f}, p={fitted_params[3]:.2f}"
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8))
    
    plt.tight_layout()
    return fig

def fit_frank_starling_curve(x, y):
    """Fit logistic curve to data with error handling"""
    if len(x) < 5:
        raise ValueError("Need at least 5 data points")
    
    # Initial parameter estimates
    initial_guess = [min(y), max(y), np.median(x), 1.0]
    
    # Parameter bounds
    bounds = (
        [min(y)-abs(max(y)-min(y)), min(y)-abs(max(y)-min(y)), min(x)*0.1, 0.1],
        [max(y)+abs(max(y)-min(y)), max(y)+abs(max(y)-min(y)), max(x)*2, 10]
    )
    
    try:
        params, _ = curve_fit(logistic_function, x, y, p0=initial_guess, bounds=bounds, maxfev=5000)
        return params
    except Exception as e:
        raise ValueError(f"Curve fitting failed: {str(e)}")

def initialize_data():
    """Initialize session state with default data"""
    if 'default_data' not in st.session_state:
        st.session_state.default_data = pd.DataFrame({
            'IV Volume (mL)': [75, 150, 200, 250, 300],
            'ΔVTI (cm)': [None, None, None, None, None]
        })

def reset_data():
    """Reset data to default values"""
    st.session_state.default_data = pd.DataFrame({
        'IV Volume (mL)': [75, 150, 200, 250, 300],
        'ΔVTI (cm)': [None, None, None, None, None]
    })

def main():
    initialize_data()
    
    # Header
    st.title("❤️ Frank-Starling Curve Analysis")
    st.markdown("Analyze the relationship between ventricular preload and cardiac output using a 4-parameter logistic model.")
    
    # Background information
    with st.expander("📚 About the Frank-Starling Mechanism"):
        st.markdown("""
        The Frank-Starling law describes how stroke volume increases with ventricular filling.
        
        **Key Parameters:**
        - **A1**: Baseline cardiac output at minimal preload
        - **A2**: Maximum cardiac output (plateau)
        - **x0**: Optimal preload volume (inflection point)
        - **p**: Curve steepness (sensitivity to preload changes)
        """)
    
    # Main layout
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📝 Data Input")
        
        # Data editor - using the fixed approach from your code
        edited_df = st.data_editor(
            st.session_state.default_data,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor",
            column_config={
                "IV Volume (mL)": st.column_config.NumberColumn(min_value=0, max_value=1000, step=10),
                "ΔVTI (cm)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
            }
        )
        
        # Data management buttons
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("🔄 Reset Data", use_container_width=True):
                reset_data()
                st.rerun()
        
        with col1b:
            csv_data = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Export CSV", csv_data, "frank_starling_data.csv", "text/csv", use_container_width=True)
        
        # File upload
        uploaded_file = st.file_uploader("Import CSV Data", type=["csv"])
        if uploaded_file:
            try:
                imported_data = pd.read_csv(uploaded_file)
                required_columns = {'IV Volume (mL)', 'ΔVTI (cm)'}
                if required_columns.issubset(imported_data.columns):
                    st.session_state.default_data = imported_data
                    st.success("✅ Data imported successfully!")
                    st.rerun()
                else:
                    st.error("❌ CSV must contain 'IV Volume (mL)' and 'ΔVTI (cm)' columns")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        # Get clean data (only rows with non-null ΔVTI values)
        clean_data = edited_df.dropna()
        x = clean_data['IV Volume (mL)'].values
        y = clean_data['ΔVTI (cm)'].values
        
        if len(x) == 0:
            st.info("📝 Enter ΔVTI values to see the analysis")
        elif len(x) < 5:
            st.warning(f"⚠️ Need at least 5 ΔVTI data points to fit the curve (currently have {len(x)})")
            st.info("Showing scatter plot of current data")
            
            # Show scatter plot for insufficient data
            fig = create_scatter_plot(x, y)
            st.pyplot(fig)
            
            # Download scatter plot
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("📥 Download Plot", buf.getvalue(), "frank_starling_scatter.png", "image/png")
            
        else:
            try:
                # Fit curve with 5+ data points
                params = fit_frank_starling_curve(x, y)
                
                # Display parameters
                st.markdown("### 📈 Curve Parameters")
                col2a, col2b, col2c, col2d = st.columns(4)
                
                with col2a:
                    st.metric("Baseline (A1)", f"{params[0]:.2f}")
                with col2b:
                    st.metric("Plateau (A2)", f"{params[1]:.2f}")
                with col2c:
                    st.metric("Optimal Preload (x0)", f"{params[2]:.1f} mL")
                with col2d:
                    st.metric("Slope (p)", f"{params[3]:.2f}")
                
                # Create and display plot
                fig = create_frank_starling_plot(x, y, params)
                st.pyplot(fig)
                
                # Download plot
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("📥 Download Plot", buf.getvalue(), "frank_starling_curve.png", "image/png")
                
                # Clinical interpretation
                st.markdown("### 🧠 Clinical Interpretation")
                cardiac_reserve = params[1] - params[0]
                preload_sensitivity = 'High' if params[3] > 1.5 else 'Moderate' if params[3] > 0.8 else 'Low'
                
                interpretation = f"""
                **Cardiac Reserve**: {cardiac_reserve:.2f} cm (difference between plateau and baseline)
                
                **Preload Sensitivity**: {preload_sensitivity} (slope factor = {params[3]:.2f})
                
                **Optimal Volume**: {params[2]:.1f} mL represents the ideal preload for maximum efficiency
                """
                
                st.markdown(interpretation)
                
                # Recommendations
                if params[3] > 2.0:
                    st.info("💡 High preload sensitivity - patient may benefit significantly from volume optimization")
                elif params[3] < 1.0:
                    st.warning("⚠️ Low preload sensitivity - limited responsiveness to volume changes")
                
                if len(x) < 8:
                    st.warning("📊 Recommendation: Add more data points (8+ recommended) for improved accuracy")
                
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                st.info("💡 Try adding more data points across a wider range of volumes")
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
