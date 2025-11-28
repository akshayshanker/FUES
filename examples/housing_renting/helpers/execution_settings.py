"""
Execution settings management for solver_runner.

This module centralizes all execution settings and runtime parameters for the 
housing-renting solver, providing a clean interface for managing paths, methods, 
metrics, and grid sizes from PBS job arguments.

Note: This is distinct from the model configuration (YAML files) which defines
the economic model itself.
"""

from pathlib import Path
import json
import numpy as np


class ExecutionSettings:
    """Centralized execution settings management for solver runner."""
    
    # Class constants - moved from solve_runner.py
    DEFAULT_BASE = "VFI_HDGRID_GPU"
    ALL_METHODS = ["VFI_HDGRID", "VFI_HDGRID_GPU", "FUES", "DCEGM", "CONSAV", "FUES2DEV"]
    DEFAULT_FAST_METHODS = ["FUES", "CONSAV", "DCEGM","FUES_V0DEV"]
    DEFAULT_COMPARISON_METRICS = "dev_c_L2,plot_c_comparison,plot_v_comparison"
    
    def __init__(self, args, cfg_dir_base, timestamp_suffix=None):
        """
        Initialize configuration manager.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments
        cfg_dir_base : Path
            Base directory for configuration files
        timestamp_suffix : str, optional
            Timestamp suffix for image directories
        """
        self.args = args
        self.cfg_dir_base = cfg_dir_base
        self.timestamp_suffix = timestamp_suffix

        # Initialize all configuration components
        self.setup_paths()
        self.setup_grid_sizes()
        self.setup_methods()
        self.setup_metrics()
        self.setup_loading_config()
        
    def setup_paths(self):
        """Configure all file system paths."""
        # Output paths
        self.packroot = Path.cwd()
        self.output_root = self.packroot / self.args.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Config directory for this bundle
        self.cfg_dir_bundle = self.cfg_dir_base / self.args.bundle_prefix

        # Image directory for plots - ALWAYS use timestamp suffix to preserve old runs
        if self.args.plots or self.args.csv_export:
            # Always create timestamped directory to preserve previous runs
            if self.timestamp_suffix:
                self.img_dir = self.output_root / f"images_{self.timestamp_suffix}"
            else:
                # Fallback: generate timestamp here if not provided
                from datetime import datetime
                fallback_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.img_dir = self.output_root / f"images_{fallback_timestamp}"
        else:
            # No plots or csv export requested, use default
            self.img_dir = self.output_root / "images"
        
    def setup_grid_sizes(self):
        """Parse and configure grid sizes."""
        self.vf_ngrid = int(float(self.args.vfi_ngrid))
        self.hd_points = int(float(self.args.HD_points))
        self.std_points = int(float(self.args.grid_points))
        self.pb_delta = float(self.args.delta_pb)
        
    def setup_methods(self):
        """Configure solution methods and determine what to run."""
        # Determine baseline method
        if self.args.baseline_method:
            self.baseline_method = self.args.baseline_method.upper()
        else:
            # Auto-detect based on GPU flag
            self.baseline_method = "VFI_HDGRID_GPU" if self.args.gpu else "VFI_HDGRID"
        
        # Parse fast methods
        if self.args.fast_methods:
            self.fast_methods = [m.strip().upper() for m in self.args.fast_methods.split(",")]
        else:
            self.fast_methods = self.DEFAULT_FAST_METHODS
        
        # Parse method list
        if self.args.ue_method.upper() == "ALL":
            self.methods_to_run = self.ALL_METHODS
        else:
            self.methods_to_run = [m.strip().upper() for m in self.args.ue_method.split(",")]
        
        # Automatically include baseline if requested and not already present
        if self.args.include_baseline and self.baseline_method not in self.methods_to_run:
            self.methods_to_run.insert(0, self.baseline_method)
        
        # Determine which fast methods to actually run
        self.fast_methods_to_run = [m for m in self.fast_methods if m in self.methods_to_run]
        
        # Determine if baseline should be run
        self.should_run_baseline = (self.baseline_method in self.methods_to_run and 
                                    self.args.include_baseline)
        
        # Determine baseline grid points based on method type
        if self.baseline_method in ["VFI_HDGRID", "VFI_HDGRID_GPU"]:
            self.baseline_points = self.hd_points
        else:
            # For fast methods used as baseline
            self.baseline_points = self.std_points
            
    def setup_metrics(self):
        """Configure metrics and comparison settings."""
        # Parse comparison metrics that require baseline
        self.comparison_metrics = set(
            m.strip() for m in self.args.comparison_metrics.split(",") if m.strip()
        )
        
        # Parse requested metrics
        if self.args.metrics.lower() == "all":
            self.requested_metrics = ["euler_error", "dev_c_L2", "plot_c_comparison", "plot_v_comparison"]
        else:
            self.requested_metrics = [m.strip() for m in self.args.metrics.split(",")]
            
            # Handle special case: "plots" adds all plot metrics
            if "plots" in self.requested_metrics:
                self.requested_metrics.remove("plots")
                # Add plot comparison metrics
                self.requested_metrics.extend(["plot_c_comparison", "plot_v_comparison"])
        
        # Determine if baseline is needed for comparisons
        self.needs_baseline = bool(
            set(self.requested_metrics) & self.comparison_metrics
        ) or self.args.include_baseline
        
    def setup_loading_config(self):
        """Configure selective model loading settings."""
        self.periods_to_load = None
        self.stages_to_load = None
        
        # Parse periods to load
        if self.args.load_periods:
            try:
                self.periods_to_load = [int(p.strip()) for p in self.args.load_periods.split(",")]
            except ValueError:
                print(f"Warning: Invalid --load-periods format, ignoring: {self.args.load_periods}")
        
        # Parse stages to load
        if self.args.load_stages:
            try:
                stages_dict = json.loads(self.args.load_stages)
                # Convert string keys to int keys
                self.stages_to_load = {int(k): v for k, v in stages_dict.items()}
            except (json.JSONDecodeError, ValueError):
                print(f"Warning: Invalid --load-stages format, ignoring: {self.args.load_stages}")
    
    def get_baseline_params(self):
        """
        Get parameter vector for baseline method.
        
        Returns
        -------
        np.ndarray
            Parameter vector for baseline configuration
        """
        return np.array([
            self.baseline_method,
            self.baseline_points,
            self.baseline_points,
            self.baseline_points,
            self.pb_delta
        ], dtype=object)
    
    def get_method_params(self, method):
        """
        Get parameter vector for a specific method.
        
        Parameters
        ----------
        method : str
            Method name
            
        Returns
        -------
        np.ndarray
            Parameter vector for the method
        """
        # Determine grid points based on method type
        if method in ["VFI_HDGRID", "VFI_HDGRID_GPU"]:
            points = self.hd_points
        else:
            points = self.std_points
            
        return np.array([
            method,
            points,
            points,
            points,
            self.pb_delta
        ], dtype=object)
    
    def get_baseline_metrics_filter(self, all_metrics):
        """
        Filter out comparison metrics for baseline computation.
        
        Parameters
        ----------
        all_metrics : dict
            All available metrics
            
        Returns
        -------
        dict
            Metrics with comparison metrics removed
        """
        return {k: v for k, v in all_metrics.items() if k not in self.comparison_metrics}
    
    def should_skip_baseline_comparison(self):
        """
        Determine if baseline comparison should be skipped.
        
        Returns
        -------
        bool
            True if only baseline is being run (no fast methods)
        """
        return self.should_run_baseline and not self.fast_methods_to_run
    
    def get_plot_config(self):
        """
        Get plotting configuration.
        
        Returns
        -------
        dict
            Configuration for plot generation
        """
        # EGM bounds for plots
        egm_bounds = {
            'value_h14': (27, 27.5, 8.220, 8.30),
            'assets_h14': (27, 27.5, 20, 24),
            'value_h0': (2, 4, -2, None),
            'assets_h0': (2,4, 0, 4),
            'value_h5': (25, 30.5, 6.9, 7),
            'assets_h5': (25, 30, 24, 26),
            'value_h3': (25, 30.5, None, None),
            'assets_h3': (25, 30, None, None),
        }
        
        # Dimension labels for policy arrays
        asset_dims = {
            0: 'w_idx',    # wealth capital (housing)
            1: 'h_idx',    # Liquid assets
            2: 'y_idx'     # The decision/choice axis
        }
        
        # Specific indices to plot
        plots_of_interest = {
            'h_idx': [5, 10, 14]  # Generate plots only for these h indices
        }
        
        return {
            'egm_bounds': egm_bounds,
            'asset_dims': asset_dims,
            'plots_of_interest': plots_of_interest,
            'y_idx_list': (0, 1, 2)  # Default y indices for plots
        }
    
    def get_memory_config(self):
        """
        Get memory management configuration.
        
        Returns
        -------
        dict
            Memory management settings
        """
        return {
            'low_memory': self.args.low_memory,
            'save_full_model': self.args.save_full_model,
            'free_memory': not self.args.save_full_model,
            'periods_to_keep': [0, 1] if not self.args.save_full_model else None
        }
    
    def print_configuration(self, is_root=True):
        """Print configuration summary."""
        if not is_root:
            return
            
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Baseline method: {self.baseline_method}")
        print(f"Fast methods: {', '.join(self.fast_methods_to_run) if self.fast_methods_to_run else 'None'}")
        print(f"Grid sizes: VFI={self.vf_ngrid}, HD={self.hd_points}, STD={self.std_points}")
        print(f"Output root: {self.output_root}")
        print(f"Bundle prefix: {self.args.bundle_prefix}")
        
        if self.periods_to_load or self.stages_to_load:
            print(f"Selective loading: periods={self.periods_to_load}, stages={self.stages_to_load}")
        
        if self.args.gpu:
            print("GPU acceleration: ENABLED")
        if self.args.mpi:
            print("MPI parallelization: ENABLED")
        if self.args.low_memory:
            print("Low memory mode: ENABLED")
        
        print("="*60 + "\n")