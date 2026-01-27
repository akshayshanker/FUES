"""
Execution settings management for solver_runner.

This module centralizes all execution settings and runtime parameters for the 
housing-renting solver, providing a clean interface for managing paths, methods, 
metrics, and grid sizes from PBS job arguments.

Note: This is distinct from the model configuration (YAML files) which defines
the economic model itself.
"""

from pathlib import Path
import numpy as np
import yaml


def load_experiment_set(experiment_set_name: str = "default") -> dict:
    """
    Load experiment set configuration from YAML file.
    
    Parameters
    ----------
    experiment_set_name : str
        Name of the experiment set (without .yml extension)
        
    Returns
    -------
    dict
        Experiment set configuration
    """
    # Look for experiment set in experiments/housing_renting/experiment_sets/
    experiment_sets_dir = Path(__file__).parent.parent.parent.parent / "experiments" / "housing_renting" / "experiment_sets"
    config_path = experiment_sets_dir / f"{experiment_set_name}.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment set '{experiment_set_name}' not found at {config_path}\n"
            f"Available sets: {[f.stem for f in experiment_sets_dir.glob('*.yml')]}"
        )
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ExecutionSettings:
    """Centralized execution settings management for solver runner."""
    
    # Class constants - moved from solve_runner.py
    DEFAULT_BASE = "VFI_HDGRID_GPU"
    ALL_METHODS = ["VFI_HDGRID", "VFI_HDGRID_GPU", "FUES", "DCEGM", "CONSAV", "FUES2DEV", "VFI"]
    DEFAULT_FAST_METHODS = ["FUES", "CONSAV", "DCEGM", "VFI"]
    DEFAULT_COMPARISON_METRICS = "dev_c_L2,plot_c_comparison,plot_v_comparison"
    
    # Default parameter paths (can be overridden by experiment set)
    DEFAULT_PARAM_PATHS = [
        "master.methods.upper_envelope",   # Method (excluded from hash)
        "master.settings.a_points",        # Grid points (asset)
        "master.settings.a_nxt_points",    # Grid points (next period asset)
        "master.settings.w_points",        # Grid points (wealth)
        "master.parameters.delta_pb",      # Price bound delta
    ]
    
    @classmethod
    def get_param_paths(cls, experiment_set: str = "default") -> list:
        """
        Get param_paths from experiment set configuration.
        
        Parameters
        ----------
        experiment_set : str
            Name of experiment set to load
            
        Returns
        -------
        list
            Parameter paths for CircuitRunner
        """
        try:
            config = load_experiment_set(experiment_set)
            return config.get("param_paths", cls.DEFAULT_PARAM_PATHS)
        except FileNotFoundError:
            print(f"Warning: Experiment set '{experiment_set}' not found, using defaults from single run config set-up")
            return cls.DEFAULT_PARAM_PATHS
    
    @classmethod
    def get_sweep_config(cls, experiment_set: str) -> dict:
        """
        Get sweep configuration from experiment set.

        Parameters
        ----------
        experiment_set : str
            Name of experiment set to load

        Returns
        -------
        dict
            Sweep configuration with:
            - 'methods', 'grid_sizes', 'H_sizes': sweep dimensions
            - 'fixed': fixed parameters (periods, vfi_ngrid, delta_pb)
            - 'param_paths': parameter paths for bundle hashing
            - 'ref_method': baseline method for comparison (e.g., 'VFI_HDGRID_GPU')
            - 'ref_params_override': overrides when building ref_params (e.g., {'grid_sizes': 200000})
            - 'config_id', 'trial_id': experiment identification
            - 'metrics': list of metrics to compute (e.g., ['euler_error', 'dev_c_L2'])
        """
        config = load_experiment_set(experiment_set)
        sweep = config.get("sweep", {})
        fixed = config.get("fixed", {})
        return {
            "methods": sweep.get("methods", ["FUES", "CONSAV", "VFI"]),
            "grid_sizes": sweep.get("grid_sizes", [500, 1000, 2000]),
            "H_sizes": sweep.get("H_sizes", [7, 10, 15]),
            "fixed": fixed,
            "param_paths": config.get("param_paths", cls.DEFAULT_PARAM_PATHS),
            # Reference params for baseline comparison
            "ref_method": config.get("ref_method", None),
            "ref_params_override": config.get("ref_params_override", {}),
            # Experiment identification
            "config_id": config.get("config_id", None),
            "trial_id": config.get("trial_id", None),
            # Metrics to compute
            "metrics": config.get("metrics", None),
        }
    
    @classmethod 
    def build_sweep_design_matrix(cls, experiment_set: str) -> tuple:
        """
        Build design matrix for parameter sweep.
        
        Parameters
        ----------
        experiment_set : str
            Name of experiment set to load
            
        Returns
        -------
        tuple
            (design_matrix as np.ndarray, param_paths list, sweep_config dict)
        """
        from itertools import product
        
        sweep_config = cls.get_sweep_config(experiment_set)
        methods = sweep_config["methods"]
        grid_sizes = sweep_config["grid_sizes"]
        H_sizes = sweep_config["H_sizes"]
        param_paths = sweep_config["param_paths"]
        
        # Build all combinations, grouped by H_points for optimal baseline cache reuse
        # The baseline hash depends on H_points (not method or grid_size due to ref_params_override)
        # Grouping by H ensures mpi_map assigns configs with same baseline to same rank
        rows = []
        for H in H_sizes:
            for grid in grid_sizes:
                for method in methods:
                    # Row order must match param_paths order:
                    # [method, a_points, H_points]
                    rows.append([method, grid, H])
        
        return np.array(rows, dtype=object), param_paths, sweep_config
    
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
        
    def setup_paths(self):
        """Configure all file system paths."""
        # Output paths
        self.packroot = Path.cwd()
        self.output_root = self.packroot / self.args.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Config directory for this configuration
        self.cfg_dir_bundle = self.cfg_dir_base / self.args.config_id

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
        """Configure metrics and comparison settings.
        
        Metrics can be specified via:
        1. Command-line: --metrics "euler_error,dev_c_L2"
        2. Experiment YAML: metrics: [euler_error, dev_c_L2]
        
        Command-line takes precedence over YAML if explicitly provided.
        """
        # Parse comparison metrics that require baseline
        self.comparison_metrics = set(
            m.strip() for m in self.args.comparison_metrics.split(",") if m.strip()
        )
        
        # Check for YAML-defined metrics (sweep mode)
        yaml_metrics = None
        if hasattr(self.args, 'experiment_set') and self.args.experiment_set:
            sweep_config = self.get_sweep_config(self.args.experiment_set)
            yaml_metrics = sweep_config.get("metrics")
        
        # Determine requested metrics (command-line overrides YAML)
        # Default command-line value is "all", so check if user explicitly set something else
        cmd_metrics = self.args.metrics.lower().strip()
        
        if cmd_metrics == "all":
            # Use YAML metrics if available, otherwise default "all" set
            if yaml_metrics:
                self.requested_metrics = yaml_metrics
            else:
                self.requested_metrics = ["euler_error", "dev_c_L2", "plot_c_comparison", "plot_v_comparison"]
        else:
            # Command-line explicitly specified - use those
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
        # Tax bracket boundaries are at a=5.00, 5.50, 7.74, 13.94
        # Add zoom windows around these to see constraint segments
        # Keys: value_y{y_idx}_h{h_idx}, assets_y{y_idx}_h{h_idx}
        egm_bounds = {
            'value_h14': (27, 27.5, 8.220, 8.30),
            'assets_h14': (27, 27.5, 20, 24),
            'value_h0': (2, 4, -2, None),
            'assets_h0': (2, 4, 0, 4),
            'value_h5': (25, 30.5, 6.9, 7),
            'assets_h5': (25, 30, 24, 26),
            'value_h3': (25, 30.5, None, None),
            'assets_h3': (25, 30, None, None),
            # Tax bracket zoom: y_idx=0 (low income), various H
            # Zoom on endogenous grid m around tax transitions
            # Value: y lower bound -5.5; Assets: y upper bound 12
            'value_y0_h0': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y0_h0': (4, 16, None, 12),
            'value_y0_h3': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y0_h3': (4, 16, None, 12),
            'value_y0_h5': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y0_h5': (4, 16, None, 12),
            # Tax bracket zoom: y_idx=24 (mid income)
            'value_y24_h0': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y24_h0': (4, 16, None, 12),
            'value_y24_h3': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y24_h3': (4, 16, None, 12),
            'value_y24_h5': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y24_h5': (4, 16, None, 12),
            # Tax bracket zoom: y_idx=48 (high income)
            'value_y48_h0': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y48_h0': (4, 16, None, 12),
            'value_y48_h3': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y48_h3': (4, 16, None, 12),
            'value_y48_h5': (4, 16, None, None),  # Auto-fit y-axis
            'assets_y48_h5': (4, 16, None, 12),
            # Wide view (0-10): y_idx=0 (low income)
            'value_y0_h0_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y0_h0_wide': (0, 10, None, 12),
            'value_y0_h3_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y0_h3_wide': (0, 10, None, 12),
            'value_y0_h5_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y0_h5_wide': (0, 10, None, 12),
            # Wide view (0-10): y_idx=24 (mid income)
            'value_y24_h0_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y24_h0_wide': (0, 10, None, 12),
            'value_y24_h3_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y24_h3_wide': (0, 10, None, 12),
            'value_y24_h5_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y24_h5_wide': (0, 10, None, 12),
            # Wide view (0-10): y_idx=48 (high income)
            'value_y48_h0_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y48_h0_wide': (0, 10, None, 12),
            'value_y48_h3_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y48_h3_wide': (0, 10, None, 12),
            'value_y48_h5_wide': (0, 10, None, None),  # Auto-fit y-axis
            'assets_y48_h5_wide': (0, 10, None, 12),
            # Zoom view (3-9, value -3 to 2.5): y_idx=0 (low income)
            'value_y0_h0_zoom': (3, 9, -3, 2.5),
            'assets_y0_h0_zoom': (3, 9, None, 12),
            'value_y0_h3_zoom': (3, 9, -3, 2.5),
            'assets_y0_h3_zoom': (3, 9, None, 12),
            'value_y0_h5_zoom': (3, 9, -3, 2.5),
            'assets_y0_h5_zoom': (3, 9, None, 12),
            # Zoom view (3-9, value -3 to 2.5): y_idx=24 (mid income)
            'value_y24_h0_zoom': (3, 9, -3, 2.5),
            'assets_y24_h0_zoom': (3, 9, None, 12),
            'value_y24_h3_zoom': (3, 9, -3, 2.5),
            'assets_y24_h3_zoom': (3, 9, None, 12),
            'value_y24_h5_zoom': (3, 9, -3, 2.5),
            'assets_y24_h5_zoom': (3, 9, None, 12),
            # Zoom view (3-9): y_idx=48 (high income)
            'value_y48_h0_zoom': (3, 9, -3.5, 0),  # value y-axis -3.5 to 0
            'assets_y48_h0_zoom': (3, 9, 1, 8),  # assets y-axis 1 to 8 (excludes 8.36 line)
            'value_y48_h3_zoom': (3, 9, -3, 2.5),
            'assets_y48_h3_zoom': (3, 9, None, 12),
            'value_y48_h5_zoom': (3, 9, -3, 2.5),
            'assets_y48_h5_zoom': (3, 9, None, 12),
            # Zoom2 view (4.5-7.5): y_idx=0 (low income)
            'value_y0_h0_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y0_h0_zoom2': (4.5, 7.5, None, 12),
            'value_y0_h3_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y0_h3_zoom2': (4.5, 7.5, None, 12),
            'value_y0_h5_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y0_h5_zoom2': (4.5, 7.5, None, 12),
            # Zoom2 view (4.5-7.5): y_idx=24 (mid income)
            'value_y24_h0_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y24_h0_zoom2': (4.5, 7.5, None, 12),
            'value_y24_h3_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y24_h3_zoom2': (4.5, 7.5, None, 12),
            'value_y24_h5_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y24_h5_zoom2': (4.5, 7.5, None, 12),
            # Zoom2 view (4.5-7.5): y_idx=48 (high income)
            'value_y48_h0_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y48_h0_zoom2': (4.5, 7.5, None, 12),
            'value_y48_h3_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y48_h3_zoom2': (4.5, 7.5, None, 12),
            'value_y48_h5_zoom2': (4.5, 7.5, None, None),  # Auto-fit y-axis
            'assets_y48_h5_zoom2': (4.5, 7.5, None, 12),
            # Kink points (tax bracket boundaries) for horizontal lines on assets panel
            'kink_points': [2.20, 2.50, 2.75, 3.87, 6.97, 8.36, 12.0, 15.0, 20.0],
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
        
        # Policy plot configuration
        # - zoom_windows: list of (x_min, x_max) tuples for asset/wealth axis
        #   Use (None, None) for full range
        # - y_idx_list: income grid indices to plot
        # - H_idx_list: housing grid indices to plot
        # Tax bracket boundaries: a=5.00, 5.50, 7.74, 13.94
        policy_config = {
            'zoom_windows': [
                (None, None),  # Full range
                (0, 5),        # Assets 0-5
                (4, 8),        # Tax bracket zoom: a=5.0 to 7.74
                (5, 10),       # Assets 5-10
                (7, 15),       # Tax bracket zoom: a=7.74 to 13.94
                (10, 15),      # Assets 10-15
                (15, 20),      # Assets 15-20
                (20, 25),      # Assets 20-25
                (25, 30),      # Assets 25-30
                (30, 35),      # Assets 30-35
                (35, 40),      # Assets 35-40
            ],
            'y_idx_list': [0, 24, 48],    # Income indices: low (0), mid (24), high (48)
            'H_idx_list': [0, 5, 10, 14], # Housing indices to plot
        }
        
        return {
            'egm_bounds': egm_bounds,
            'asset_dims': asset_dims,
            'plots_of_interest': plots_of_interest,
            'y_idx_list': [0, 24, 48],  # EGM plots: low (0), mid (24), high (48) income indices
            'policy_config': policy_config,
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
        print(f"Config ID: {self.args.config_id}")
        
        if self.args.gpu:
            print("GPU acceleration: ENABLED")
        if self.args.mpi:
            print("MPI parallelization: ENABLED")
        if self.args.low_memory:
            print("Low memory mode: ENABLED")
        
        print("="*60 + "\n")