"""
Automatic Windowing Parameter Calculation
==========================================

Calculate optimal window center and window width values from image histograms.

This module is particularly useful for datasets where DICOM windowing metadata
has been stripped during anonymization (e.g., InBreast dataset).

Usage:
    from calculate_windowing import calculate_windowing, analyze_dataset_windowing
    
    # Single image
    window_center, window_width = calculate_windowing(image, method='breast_tissue')
    
    # Multiple images (recommended)
    stats = analyze_dataset_windowing(dicom_dir, num_samples=50)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json

def _first_numeric(value) -> Optional[float]:
    """Return first numeric value from scalar or multi-value DICOM field."""
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return float(value[0])

    # pydicom MultiValue supports indexing
    if hasattr(value, '__len__') and hasattr(value, '__getitem__') and not isinstance(value, (str, bytes)):
        try:
            return float(value[0])
        except Exception:
            pass

    return float(value)

def get_dicom_voi_lut_params(dcm) -> Optional[Dict[str, float]]:
    """
    Extract VOI LUT parameters from a pydicom dataset.

    Returns None when required tags are missing or invalid.
    """
    center = _first_numeric(getattr(dcm, 'WindowCenter', None))
    width = _first_numeric(getattr(dcm, 'WindowWidth', None))

    if center is None or width is None or width <= 0:
        print("Warning: DICOM VOI LUT parameters missing or invalid. Falling back to histogram-based windowing.")
        center, width = calculate_windowing(dcm.pixel_array, 
                                            'breast_tissue',
                                            exclude_background= True)
    
    intercept = _first_numeric(getattr(dcm, 'RescaleIntercept', None))
    slope = _first_numeric(getattr(dcm, 'RescaleSlope', None))
    voi_func = getattr(dcm, 'VOILUTFunction', None)

    return {
        'window_center': int(round(center)),
        'window_width': int(round(width)),
        'rescale_intercept': 0.0 if intercept is None else float(intercept),
        'rescale_slope': 1.0 if slope is None else float(slope),
        'voi_lut_function': str(voi_func) if voi_func is not None else 'LINEAR',
    }

def should_invert_monochrome1(dicom_dataset) -> bool:
    """Return True when DICOM photometric interpretation is MONOCHROME1."""
    photometric = str(getattr(dicom_dataset, 'PhotometricInterpretation', '')).upper()
    ## Different DICOM tags may indicate inversion is needed.
    # MONOCHROME1 is the standard, but ....
    # (0028,0004) Photometric Interpretation: MONOCHROME1 or MONOCHROME2
    # (0008,0068) PresentationIntentType:  FOR PROCESSING o FOR PRESENTATION
    # (2050,0020) PresentationLUTShape:  INVERSE or IDENTITY
    return photometric == 'MONOCHROME1'

def normalize_photometric(image: np.ndarray, dicom_dataset) -> Tuple[np.ndarray, bool]:
    """
    Normalize MONOCHROME1 images to MONOCHROME2-like intensity ordering.

    In MONOCHROME1, lower pixel values are displayed brighter; we invert intensities
    so downstream preprocessing behaves consistently.
    """
    if not should_invert_monochrome1(dicom_dataset):
        return image, False

    image_min = np.min(image)
    image_max = np.max(image)
    inverted = (image_max + image_min) - image
    return inverted, True


def get_windowing_for_preprocessing(image: np.ndarray,
                                    dicom_dataset=None,
                                    histogram_method: str = 'breast_tissue',
                                    exclude_background: bool = True,
                                    prefer_dicom: bool = True) -> Dict[str, object]:
    """
    Resolve windowing parameters for preprocessing.

    Priority:
    1) Use DICOM VOI LUT when available.
    2) Fallback to histogram-based estimation.

    Returns a dictionary with windowing parameters and provenance metadata.
    """
    if prefer_dicom and dicom_dataset is not None:
        dicom_params = get_dicom_voi_lut_params(dicom_dataset)
        if dicom_params is not None:
            return {
                'window_center': dicom_params['window_center'],
                'window_width': dicom_params['window_width'],
                'source': 'dicom_voi_lut',
                'method_used': 'dicom_voi_lut',
                'rescale_intercept': dicom_params['rescale_intercept'],
                'rescale_slope': dicom_params['rescale_slope'],
                'voi_lut_function': dicom_params['voi_lut_function'],
            }

    center, width = calculate_windowing(
        image=image,
        method=histogram_method,
        exclude_background=exclude_background,
    )

    return {
        'window_center': center,
        'window_width': width,
        'source': 'histogram_fallback',
        'method_used': histogram_method,
        'rescale_intercept': 0.0,
        'rescale_slope': 1.0,
        'voi_lut_function': 'LINEAR',
    }


def get_windowing_for_dicom_path(dicom_path: Path,
                                 histogram_method: str = 'breast_tissue',
                                 exclude_background: bool = True,
                                 prefer_dicom: bool = True,
                                 normalize_monochrome1: bool = True) -> Dict[str, object]:
    """
    Convenience wrapper: load DICOM and resolve windowing for preprocessing.
    """
    import pydicom

    dcm = pydicom.dcmread(dicom_path, stop_before_pixels=False)
    image = dcm.pixel_array
    inversion_recommended = should_invert_monochrome1(dcm)
    inversion_applied = False

    if normalize_monochrome1:
        image, inversion_applied = normalize_photometric(image, dcm)

    result = get_windowing_for_preprocessing(
        image=image,
        dicom_dataset=dcm,
        histogram_method=histogram_method,
        exclude_background=exclude_background,
        prefer_dicom=prefer_dicom,
    )

    result['photometric_interpretation'] = str(
        getattr(dcm, 'PhotometricInterpretation', 'MISSING')
    )
    result['monochrome1_inversion_recommended'] = inversion_recommended
    result['monochrome1_inversion_applied'] = inversion_applied

    return result


def calculate_windowing(image: np.ndarray, 
                       method: str = 'percentile_2_98',
                       exclude_background: bool = True) -> Tuple[int, int]:
    """
    Calculate optimal window center and width from image histogram.
    
    Args:
        image: 2D numpy array of pixel values
        method: Calculation method. Options:
            - 'percentile_1_99': Percentiles 1-99 (robust, general use)
            - 'percentile_2_98': Percentiles 2-98 (more conservative)
            - 'percentile_5_95': Percentiles 5-95 (tight, high contrast)
            - 'breast_tissue': Optimized for mammography breast tissue (25-95 percentile)
            - 'statistical': Mean ± 2 standard deviations
            - 'statistical_wide': Mean ± 3 standard deviations
            - 'full_range': Complete value range (min-max)
            - 'histogram_peak': Based on histogram mode and FWHM
        exclude_background: If True, exclude zero pixels (background)
    
    Returns:
        Tuple of (window_center, window_width) as integers
    
    Example:
        >>> import pydicom
        >>> dcm = pydicom.dcmread('mammogram.dcm')
        >>> center, width = calculate_windowing(dcm.pixel_array, method='breast_tissue')
        >>> print(f"Window: {center}/{width}")
    """
    # Filter background if requested
    if exclude_background:
        tissue_pixels = image[image > 0]
        if len(tissue_pixels) == 0:
            raise ValueError("No non-zero pixels found in image")
    else:
        tissue_pixels = image.flatten()
    
    # Calculate based on method
    if method == 'percentile_1_99':
        p1, p99 = np.percentile(tissue_pixels, [1, 99])
        center = (p1 + p99) / 2
        width = p99 - p1
        
    elif method == 'percentile_2_98':
        p2, p98 = np.percentile(tissue_pixels, [2, 98])
        center = (p2 + p98) / 2
        width = p98 - p2
        
    elif method == 'percentile_5_95':
        p5, p95 = np.percentile(tissue_pixels, [5, 95])
        center = (p5 + p95) / 2
        width = p95 - p5
        
    elif method == 'breast_tissue':
        # Optimized for mammography: focus on glandular tissue
        # Lower percentile to include fatty tissue, higher for dense tissue
        p25, p95 = np.percentile(tissue_pixels, [25, 95])
        center = (p25 + p95) / 2
        width = (p95 - p25) * 1.5  # Expand slightly for better visualization
        
    elif method == 'statistical':
        mean = np.mean(tissue_pixels)
        std = np.std(tissue_pixels)
        center = mean
        width = 4 * std  # ±2 standard deviations
        
    elif method == 'statistical_wide':
        mean = np.mean(tissue_pixels)
        std = np.std(tissue_pixels)
        center = mean
        width = 6 * std  # ±3 standard deviations
        
    elif method == 'full_range':
        min_val = np.min(tissue_pixels)
        max_val = np.max(tissue_pixels)
        center = (min_val + max_val) / 2
        width = max_val - min_val
        
    elif method == 'histogram_peak':
        # Calculate histogram
        hist, bins = np.histogram(tissue_pixels, bins=256)
        
        # Find peak (mode)
        peak_idx = np.argmax(hist)
        peak_value = bins[peak_idx]
        center = peak_value
        
        # Find Full Width at Half Maximum (FWHM)
        half_max = hist[peak_idx] / 2
        
        # Find left side of FWHM
        left_idx = peak_idx
        while left_idx > 0 and hist[left_idx] > half_max:
            left_idx -= 1
        
        # Find right side of FWHM
        right_idx = peak_idx
        while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
            right_idx += 1
        
        # Calculate width from FWHM
        left_value = bins[left_idx]
        right_value = bins[right_idx]
        width = (right_value - left_value) * 1.75 # Expand beyond FWHM
        
        # Ensure reasonable width
        if width < 100:
            width = 4 * np.std(tissue_pixels)
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid options: "
            "'percentile_1_99', 'percentile_2_98', 'percentile_5_95', "
            "'breast_tissue', 'statistical', 'statistical_wide', "
            "'full_range', 'histogram_peak'"
        )
    
    return int(center), int(width)


def calculate_all_methods(image: np.ndarray, 
                         exclude_background: bool = True) -> Dict[str, Tuple[int, int]]:
    """
    Calculate windowing parameters using all available methods.
    
    Useful for comparing different approaches and finding the best one
    for your specific dataset.
    
    Args:
        image: 2D numpy array of pixel values
        exclude_background: If True, exclude zero pixels
    
    Returns:
        Dictionary mapping method names to (center, width) tuples
    """
    methods = [
        'percentile_1_99',
        'percentile_2_98', 
        'percentile_5_95',
        'breast_tissue',
        'statistical',
        'statistical_wide',
        'full_range',
        'histogram_peak'
    ]
    
    results = {}
    for method in methods:
        try:
            results[method] = calculate_windowing(image, method, exclude_background)
        except Exception as e:
            results[method] = (None, None)
            print(f"Warning: Method '{method}' failed: {e}")
    
    return results


def analyze_dataset_windowing(image_paths: List[Path],
                              method: str = 'breast_tissue',
                              num_samples: Optional[int] = None,
                              exclude_background: bool = True) -> Dict:
    """
    Analyze windowing parameters across multiple images in a dataset.
    
    This function processes multiple images and calculates average windowing
    parameters, which is more robust than using a single image.
    
    Args:
        image_paths: List of paths to DICOM or numpy array files
        method: Calculation method (see calculate_windowing for options)
        num_samples: If provided, only analyze this many random images
        exclude_background: If True, exclude zero pixels
    
    Returns:
        Dictionary with statistics:
            - 'method': Method used
            - 'num_images': Number of images analyzed
            - 'avg_center': Average window center
            - 'avg_width': Average window width
            - 'std_center': Standard deviation of centers
            - 'std_width': Standard deviation of widths
            - 'median_center': Median window center
            - 'median_width': Median window width
            - 'recommended_center': Recommended center (median)
            - 'recommended_width': Recommended width (median)
    """
    import pydicom
    
    # Limit samples if requested
    if num_samples and len(image_paths) > num_samples:
        import random
        image_paths = random.sample(image_paths, num_samples)
    
    centers = []
    widths = []
    
    print(f"Analyzing {len(image_paths)} images...")
    
    for i, filepath in enumerate(image_paths):
        try:
            # Try to read as DICOM
            # if str(filepath).endswith('.dcm') or :
            dcm = pydicom.dcmread(filepath, stop_before_pixels=False)
            image = dcm.pixel_array
            #else:
            #     # Try as numpy array
            #    image = np.load(filepath)
            
            center, width = calculate_windowing(image, method, exclude_background)
            centers.append(center)
            widths.append(width)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")
            continue
    
    if len(centers) == 0:
        raise ValueError("No images could be processed successfully")
    
    centers = np.array(centers)
    widths = np.array(widths)
    
    results = {
        'method': method,
        'num_images': len(centers),
        'avg_center': int(np.mean(centers)),
        'avg_width': int(np.mean(widths)),
        'std_center': int(np.std(centers)),
        'std_width': int(np.std(widths)),
        'median_center': int(np.median(centers)),
        'median_width': int(np.median(widths)),
        'min_center': int(np.min(centers)),
        'max_center': int(np.max(centers)),
        'min_width': int(np.min(widths)),
        'max_width': int(np.max(widths)),
    }
    
    # Recommended values (use median for robustness)
    results['recommended_center'] = results['median_center']
    results['recommended_width'] = results['median_width']
    
    print(f"\n✓ Analysis complete: {len(centers)} images processed")
    
    return results


def compare_methods_on_dataset(image_paths: List[Path],
                               num_samples: Optional[int] = 50) -> Dict:
    """
    Compare all windowing calculation methods on a dataset.
    
    Args:
        image_paths: List of paths to DICOM files
        num_samples: Number of images to sample
    
    Returns:
        Dictionary with results for each method
    """
    methods = [
        'percentile_1_99',
        'percentile_2_98',
        'percentile_5_95',
        'breast_tissue',
        'statistical',
        'statistical_wide',
        'full_range',
        'histogram_peak'
    ]
    
    results = {}
    for method in methods:
        print(f"\nAnalyzing with method: {method}")
        try:
            results[method] = analyze_dataset_windowing(
                image_paths, 
                method=method, 
                num_samples=num_samples
            )
        except Exception as e:
            print(f"  Error: {e}")
            results[method] = None
    
    return results


def print_windowing_report(stats: Dict, title: str = "Windowing Analysis"):
    """
    Print a formatted report of windowing statistics.
    
    Args:
        stats: Statistics dictionary from analyze_dataset_windowing
        title: Report title
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"\nMethod: {stats['method']}")
    print(f"Images analyzed: {stats['num_images']}")
    print(f"\nWindow Center:")
    print(f"  Average:  {stats['avg_center']} ± {stats['std_center']}")
    print(f"  Median:   {stats['median_center']}")
    print(f"  Range:    {stats['min_center']} - {stats['max_center']}")
    print(f"\nWindow Width:")
    print(f"  Average:  {stats['avg_width']} ± {stats['std_width']}")
    print(f"  Median:   {stats['median_width']}")
    print(f"  Range:    {stats['min_width']} - {stats['max_width']}")
    print(f"\n✓ RECOMMENDED VALUES:")
    print(f"  Window Center: {stats['recommended_center']}")
    print(f"  Window Width:  {stats['recommended_width']}")
    print("=" * 70)


def save_windowing_report(stats: Dict, output_path: Path):
    """
    Save windowing statistics to JSON file.
    
    Args:
        stats: Statistics dictionary
        output_path: Path to output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Report saved to: {output_path}")


# Mammography-specific presets
MAMMOGRAPHY_PRESETS = {
    'standard': {
        'method': 'percentile_2_98',
        'description': 'Standard mammography windowing (robust)'
    },
    'high_contrast': {
        'method': 'percentile_5_95',
        'description': 'High contrast for dense tissue'
    },
    'wide_latitude': {
        'method': 'percentile_1_99',
        'description': 'Wide latitude for mixed density'
    },
    'breast_optimized': {
        'method': 'breast_tissue',
        'description': 'Optimized for breast tissue visualization'
    },
}


def get_mammography_preset(preset_name: str = 'breast_optimized') -> str:
    """
    Get a recommended preset method for mammography.
    
    Args:
        preset_name: Name of preset ('standard', 'high_contrast', 
                    'wide_latitude', 'breast_optimized')
    
    Returns:
        Method string to use with calculate_windowing
    """
    if preset_name not in MAMMOGRAPHY_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(MAMMOGRAPHY_PRESETS.keys())}"
        )
    
    preset = MAMMOGRAPHY_PRESETS[preset_name]
    print(f"Using preset: {preset_name} - {preset['description']}")
    return preset['method']


if __name__ == '__main__':
    """
    Example usage and testing.
    """
    print("Windowing Parameter Calculator")
    print("=" * 70)
    
    # Example: Analyze InBreast dataset
    inbreast_dir = Path("/home/eloygarcia/Escritorio/Datasets/INbreast/AllDICOMs")
    
    if inbreast_dir.exists():
        print(f"\nFound InBreast dataset at: {inbreast_dir}")
        dicom_files = list(inbreast_dir.glob("*.dcm"))[:50]  # Sample 50 images
        
        print(f"\nAnalyzing {len(dicom_files)} images...")
        
        # Use breast-optimized method
        method = get_mammography_preset('breast_optimized')
        stats = analyze_dataset_windowing(dicom_files, method=method)
        
        # Print report
        print_windowing_report(stats, "InBreast Dataset - Windowing Analysis")
        
        # Save report
        output_path = Path("inbreast_windowing_analysis.json")
        save_windowing_report(stats, output_path)
        
    else:
        print("\nInBreast dataset not found. Using synthetic example...")
        # Create synthetic mammography-like image
        synthetic_image = np.random.poisson(1000, size=(2048, 2048)).astype(np.uint16)
        
        # Add breast-like intensity distribution
        y, x = np.ogrid[:2048, :2048]
        mask = ((x - 1024)**2 + (y - 1024)**2) < 800**2
        synthetic_image[mask] += 500
        
        print("\nCalculating windowing parameters...")
        results = calculate_all_methods(synthetic_image)
        
        print("\nResults from all methods:")
        print("-" * 70)
        for method, (center, width) in results.items():
            if center is not None:
                print(f"{method:20s} | Center: {center:5d} | Width: {width:5d}")
