"""
Stress-Strain Curve Analysis

This script processes stress-strain curve data from finite element simulation
to extract key mechanical properties including Young's modulus and yield strength
using the 0.2% offset method. The analysis includes data preprocessing,
curve fitting, and visualization of results.

Requirements:
- scipy, pandas, numpy, matplotlib, prettytable
"""

import os
import scipy.io
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# =============================================================================
# CONSTANTS AND CONFIGURATION PARAMETERS
# =============================================================================
SPECIMEN_LENGTH = 6.0  # Original specimen length for normalization (mm)
INITIAL_FIT_POINTS = 350  # Number of initial points for elastic modulus calculation
SLIDING_WINDOW_SIZE = 50  # Window size for linear regression
CURVE_LENGTH = 1300  # Number of points to display in fitted curves
OUTPUT_FILENAME = 'mechanical_properties.xlsx'  # Results output file
MATLAB_DATA_FILE = 'curves.mat'  # Input MATLAB data file


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def calculate_elastic_modulus(stress_strain_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate Young's modulus (E) using sliding window linear regression.
    
    Identifies the region with maximum slope in the initial elastic portion
    of the stress-strain curve.
    
    Parameters:
    -----------
    stress_strain_data : pd.DataFrame
        DataFrame containing 'strain' and 'stress' columns
        
    Returns:
    --------
    Tuple[float, float]
        Young's modulus (E) and y-intercept of the fitted line
    """
    modulus_candidates = []
    
    for start_idx in range(INITIAL_FIT_POINTS):
        # Extract window for linear regression
        strain_window = stress_strain_data['strain'].values[start_idx:start_idx + SLIDING_WINDOW_SIZE]
        stress_window = stress_strain_data['stress'].values[start_idx:start_idx + SLIDING_WINDOW_SIZE]
        
        # Perform linear regression
        youngs_modulus, intercept = np.polyfit(strain_window, stress_window, 1)
        modulus_candidates.append([youngs_modulus, intercept])
    
    # Convert to DataFrame for easier processing
    modulus_df = pd.DataFrame(modulus_candidates, columns=['E', 'intercept'])
    
    # Select the maximum slope as the true elastic modulus
    max_modulus_idx = modulus_df['E'].argmax()
    optimal_modulus = modulus_df.iloc[max_modulus_idx]
    
    return optimal_modulus['E'], optimal_modulus['intercept']


def calculate_yield_strength(stress_strain_data: pd.DataFrame, 
                           offset_strain: float = 0.002) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Determine yield strength using the 0.2% offset method.
    
    Constructs a parallel line offset by 0.2% strain and finds the intersection
    with the original stress-strain curve.
    
    Parameters:
    -----------
    stress_strain_data : pd.DataFrame
        DataFrame containing 'strain' and 'stress' columns
    offset_strain : float, optional
        Strain offset for yield strength determination (default: 0.002)
        
    Returns:
    --------
    Tuple[float, float, float, np.ndarray, np.ndarray]
        yield_strength, youngs_modulus, intercept, strain_data, stress_data
    """
    strain = stress_strain_data['strain']
    stress = stress_strain_data['stress']
    
    # Calculate elastic modulus from the linear region
    youngs_modulus, intercept = calculate_elastic_modulus(stress_strain_data)
    
    # Create offset line (parallel to elastic region but shifted by offset_strain)
    offset_df = stress_strain_data.copy()
    offset_df["offset_stress"] = youngs_modulus * (strain - offset_strain) + intercept
    offset_df["stress_difference"] = offset_df["offset_stress"] - offset_df['stress']
    
    # Find intersection point (minimum absolute difference)
    offset_df = offset_df.reset_index(drop=True)
    yield_index = offset_df["stress_difference"].abs().idxmin()
    yield_strength = offset_df.iloc[yield_index]['stress']
    
    return yield_strength, youngs_modulus, intercept, strain.values, stress.values


def refine_curve_resolution(original_curve: np.ndarray) -> np.ndarray:
    """
    Enhance curve resolution by interpolating intermediate points.
    
    Applies multiple passes of midpoint interpolation to increase
    the density of data points along the curve.
    
    Parameters:
    -----------
    original_curve : np.ndarray
        Original curve data points (N x 2 array)
        
    Returns:
    --------
    np.ndarray
        Refined curve with increased point density
    """
    refined_curve = original_curve.copy()
    
    # Apply three passes of midpoint interpolation
    for _ in range(3):
        interpolated_points = []
        for i in range(len(refined_curve) - 1):
            # Keep original point
            interpolated_points.append(refined_curve[i])
            # Add midpoint between current and next point
            midpoint = (refined_curve[i] + refined_curve[i + 1]) / 2
            interpolated_points.append(midpoint)
        interpolated_points.append(refined_curve[-1])
        refined_curve = np.array(interpolated_points)
    
    return refined_curve.reshape(-1, 2)


def load_and_preprocess_data() -> Tuple[List[pd.DataFrame], List[int]]:
    """
    Load MATLAB data and preprocess stress-strain curves.
    
    Reads MATLAB file, normalizes data using specimen dimensions,
    and refines curve resolution.
    
    Returns:
    --------
    Tuple[List[pd.DataFrame], List[int]]
        List of processed DataFrames and indices of invalid curves
    """
    # Load MATLAB data file
    matlab_data = scipy.io.loadmat(MATLAB_DATA_FILE)
    raw_curves = matlab_data['curve']
    
    processed_curves = []
    invalid_indices = []
    
    for idx, curve in enumerate(raw_curves[0]):
        try:
            # Normalize strain and stress using specimen dimensions
            curve[:, 0] /= SPECIMEN_LENGTH  # Normalize strain
            curve[:, 1] /= SPECIMEN_LENGTH**2  # Normalize stress
            
            # Enhance curve resolution
            refined_curve = refine_curve_resolution(curve)
            
            # Convert to DataFrame
            curve_df = pd.DataFrame(refined_curve, columns=['strain', 'stress'])
            processed_curves.append(curve_df)
            
        except (IndexError, ValueError) as e:
            print(f"Warning: Failed to process curve {idx}: {e}")
            invalid_indices.append(idx)
    
    return processed_curves, invalid_indices


def visualize_curve_analysis(curve_index: int, 
                           stress_strain_data: pd.DataFrame,
                           yield_strength: float,
                           youngs_modulus: float,
                           intercept: float) -> None:
    """
    Generate comprehensive visualization of stress-strain analysis.
    
    Parameters:
    -----------
    curve_index : int
        Identifier for the current curve
    stress_strain_data : pd.DataFrame
        Processed stress-strain data
    yield_strength : float
        Calculated yield strength
    youngs_modulus : float
        Calculated Young's modulus
    intercept : float
        Y-intercept of the elastic region
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original stress-strain curve
    plt.plot(stress_strain_data['strain'], stress_strain_data['stress'], 
             'b-', linewidth=2, label='FEM Curve')
    
    # Generate and plot elastic line
    strain_elastic = stress_strain_data['strain'].values[:CURVE_LENGTH]
    stress_elastic = youngs_modulus * strain_elastic + intercept
    plt.plot(strain_elastic, stress_elastic, 'r--', linewidth=1.5, 
             label='Elastic Region Fit')
    
    # Generate and plot offset line
    strain_offset = strain_elastic
    stress_offset = youngs_modulus * (strain_offset - 0.002) + intercept
    plt.plot(strain_offset, stress_offset, 'g--', linewidth=1.5, 
             label='0.2% Offset Line')
    
    # Mark yield strength
    plt.axhline(y=yield_strength, color='m', linestyle=':', 
                label=f'Yield Strength: {yield_strength:.2f}')
    
    plt.xlabel('Strain', fontsize=12)
    plt.ylabel('Stress (MPa)', fontsize=12)
    plt.title(f'Stress-Strain Curve Analysis - Sample {curve_index + 1}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Main execution function for mechanical properties analysis.
    """
    print("=== Mechanical Properties Analysis ===")
    print(f"Input data: {MATLAB_DATA_FILE}")
    print(f"Output file: {OUTPUT_FILENAME}")
    print("-" * 50)
    
    # Load and preprocess experimental data
    print("Loading and preprocessing stress-strain curves...")
    processed_curves, failed_curves = load_and_preprocess_data()
    
    if failed_curves:
        print(f"Warning: {len(failed_curves)} curves could not be processed")
    
    print(f"Successfully processed {len(processed_curves)} curves")
    
    # Initialize results storage
    elastic_moduli = []
    yield_strengths = []
    combined_data = pd.DataFrame()
    
    # Process each stress-strain curve
    print("\nAnalyzing mechanical properties...")
    for idx, curve_data in enumerate(processed_curves):
        print(f"Processing curve {idx + 1}/{len(processed_curves)}...")
        
        # Clean data and calculate properties
        clean_data = curve_data.dropna().reset_index(drop=True)
        
        # Calculate yield strength and elastic modulus
        yield_strength, youngs_modulus, intercept, strain_vals, stress_vals = \
            calculate_yield_strength(clean_data)
        
        # Store results
        elastic_moduli.append(youngs_modulus)
        yield_strengths.append(yield_strength)
        
        # Generate visualization
        visualize_curve_analysis(idx, clean_data, yield_strength, 
                               youngs_modulus, intercept)
    
    # Compile results into structured format
    results_table = pd.DataFrame({
        'Youngs_Modulus': elastic_moduli,
        'Yield_Strength': yield_strengths
    })
    
    # Display formatted results
    display_table = PrettyTable()
    display_table.field_names = ["Sample", "Young's Modulus", "Yield Strength"]
    display_table.float_format = ".3"
    
    for idx, (modulus, strength) in enumerate(zip(elastic_moduli, yield_strengths)):
        display_table.add_row([idx + 1, modulus, strength])
    
    print("\n" + "=" * 60)
    print("MECHANICAL PROPERTIES SUMMARY")
    print("=" * 60)
    print(display_table)
    
    # Calculate and display statistical summary
    print(f"\nStatistical Summary:")
    print(f"Young's Modulus: {np.mean(elastic_moduli):.2f} ± {np.std(elastic_moduli):.2f}")
    print(f"Yield Strength:  {np.mean(yield_strengths):.2f} ± {np.std(yield_strengths):.2f}")
    
    # Save results to Excel file
    results_table.to_excel(OUTPUT_FILENAME, index=False)
    print(f"\nResults saved to: {OUTPUT_FILENAME}")
    
    # Display plots
    plt.show()


if __name__ == "__main__":
    main()