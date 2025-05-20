#!/usr/bin/env python
"""
Wrapper script to run param_sweep.py from the project root.
Run it with: python run_param_sweep.py --ue-method "FUES"
"""

import sys
from experiments.housing_renting.param_sweep import main

if __name__ == "__main__":
    main(sys.argv[1:])  # Pass command line arguments to main 