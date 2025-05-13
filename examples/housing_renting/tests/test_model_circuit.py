#!/usr/bin/env python3
"""
Housing Renting Model Demo

This demo implements a discrete-choice housing model where agents can either 
rent or own housing, based on the model described in the documentation.

Key features:
1. Discrete choice between renting and owning housing
2. Consumption and saving decisions
3. Housing adjustment decisions for owners
4. Housing service decisions for renters
5. Multiple connected stages representing different decision points
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add the repository root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, repo_root)

from src.stagecraft.model_circuit import ModelCircuit
from src.stagecraft.period import Period
from src.stagecraft.stage import Stage

# Define the housing model stages
class TenureChoiceStage(Stage):
    """Stage for choosing between renting and owning"""
    
    def __init__(self, name="tenure_choice"):
        super().__init__(name, decision_type="discrete")
        self.stage_type = "tenure_choice"

class OwnerHousingStage(Stage):
    """Stage for home owner's housing choice"""
    
    def __init__(self, name="owner_housing"):
        super().__init__(name, decision_type="discrete")
        self.stage_type = "owner_housing"
        self.housing_grid = [0, 1, 2, 3]  # Housing size options
        self.tau = 0.05  # Transaction cost for housing adjustment

class OwnerConsumptionStage(Stage):
    """Stage for home owner's consumption choice"""
    
    def __init__(self, name="owner_consumption"):
        super().__init__(name)
        self.stage_type = "owner_consumption"
        self.alpha = 0.7  # Utility weight on non-durable consumption
        self.kappa = 1.0  # Housing utility parameter
        self.iota = 0.1   # Housing utility parameter

class RenterHousingStage(Stage):
    """Stage for renter's housing service choice"""
    
    def __init__(self, name="renter_housing"):
        super().__init__(name, decision_type="discrete")
        self.stage_type = "renter_housing"
        self.housing_grid = [0, 1, 2, 3]  # Housing service options
        self.rental_price = 0.2  # Price per unit of housing service

class RenterConsumptionStage(Stage):
    """Stage for renter's consumption choice"""
    
    def __init__(self, name="renter_consumption"):
        super().__init__(name)
        self.stage_type = "renter_consumption"
        self.alpha = 0.7  # Utility weight on non-durable consumption
        self.kappa = 1.0  # Housing utility parameter
        self.iota = 0.1   # Housing utility parameter

def main():
    """
    Create a model for the housing rental choice problem.
    
    The model has five stages:
    1. Tenure Choice (TENU): Choose between renting and owning
    2. Owner Housing (OWNH): Choose housing stock level if owner
    3. Renter Housing (RNTH): Choose housing service level if renter
    4. Owner Consumption (OWNC): Choose consumption if owner
    5. Renter Consumption (RNTC): Choose consumption if renter
    
    The stages are connected to form a complete decision sequence.
    """
    # Create a new model circuit
    model = ModelCircuit(name="HousingRentalModel")
    
    # Number of periods (e.g., lifecycle phases)
    num_periods = 3
    
    # Create periods and stages for the lifecycle model
    for t in range(num_periods):
        period = Period(time_index=t)
        
        # Create the five stages for the period
        tenure_choice = TenureChoiceStage(name=f"tenure_choice_t{t}")
        owner_housing = OwnerHousingStage(name=f"owner_housing_t{t}")
        owner_consumption = OwnerConsumptionStage(name=f"owner_consumption_t{t}")
        renter_housing = RenterHousingStage(name=f"renter_housing_t{t}")
        renter_consumption = RenterConsumptionStage(name=f"renter_consumption_t{t}")
        
        # Add stages to the period
        period.add_stage("tenure_choice", tenure_choice)
        period.add_stage("owner_housing", owner_housing)
        period.add_stage("owner_consumption", owner_consumption)
        period.add_stage("renter_housing", renter_housing)
        period.add_stage("renter_consumption", renter_consumption)
        
        # Set up intra-period connections
        
        # From tenure choice to owner housing or renter housing
        period.connect_fwd(
            src="tenure_choice",
            tgt="owner_housing",
            branch_key="own_path"
        )
        
        period.connect_fwd(
            src="tenure_choice",
            tgt="renter_housing",
            branch_key="rent_path"
        )
        
        # From housing choices to consumption choices
        period.connect_fwd(
            src="owner_housing",
            tgt="owner_consumption",
            branch_key="owner_path"
        )
        
        period.connect_fwd(
            src="renter_housing",
            tgt="renter_consumption",
            branch_key="renter_path"
        )
        
        # Create transpose connections manually instead of using auto-generation
        # Backward connection: owner_housing -> tenure_choice
        period.connect_bwd(
            src="owner_housing",
            tgt="tenure_choice",
            branch_key="from_owner_housing"  # Branch key needed as tenure_choice has multiple incoming edges
        )
        
        # Backward connection: renter_housing -> tenure_choice
        period.connect_bwd(
            src="renter_housing",
            tgt="tenure_choice",
            branch_key="from_renter_housing"  # Branch key needed as tenure_choice has multiple incoming edges
        )
        
        # Backward connection: owner_consumption -> owner_housing
        period.connect_bwd(
            src="owner_consumption",
            tgt="owner_housing"
            # No branch key needed as owner_housing only receives from owner_consumption
        )
        
        # Backward connection: renter_consumption -> renter_housing
        period.connect_bwd(
            src="renter_consumption",
            tgt="renter_housing"
            # No branch key needed as renter_housing only receives from renter_consumption
        )
        
        # Add period to model sequence
        model.add_period(period)
    
    # Add inter-period connections
    # The consumption decisions in one period connect to the tenure choice in the next
    for t in range(num_periods - 1):
        # Connect owner's consumption to next period's tenure choice
        model.add_inter_period_connection(
            source_period=model.get_period(t),
            target_period=model.get_period(t+1),
            source_stage=model.get_period(t).get_stage("owner_consumption"),
            target_stage=model.get_period(t+1).get_stage("tenure_choice"),
            source_perch_attr="cntn",
            target_perch_attr="arvl",
            branch_key="from_owner",  # Branch key needed as tenure_choice has multiple incoming edges
            create_transpose=False  # Don't auto-create transpose, we'll do it manually
        )
        
        # Connect renter's consumption to next period's tenure choice
        model.add_inter_period_connection(
            source_period=model.get_period(t),
            target_period=model.get_period(t+1),
            source_stage=model.get_period(t).get_stage("renter_consumption"),
            target_stage=model.get_period(t+1).get_stage("tenure_choice"),
            source_perch_attr="cntn",
            target_perch_attr="arvl",
            branch_key="from_renter",  # Branch key needed as tenure_choice has multiple incoming edges
            create_transpose=False  # Don't auto-create transpose, we'll do it manually
        )
        
        # Manually create backward inter-period connections
        # Connect tenure_choice back to owner_consumption in previous period
        model.add_inter_period_connection(
            source_period=model.get_period(t+1),
            target_period=model.get_period(t),
            source_stage=model.get_period(t+1).get_stage("tenure_choice"),
            target_stage=model.get_period(t).get_stage("owner_consumption"),
            source_perch_attr="arvl",
            target_perch_attr="cntn",
            branch_key="to_owner_consumption",  # Branch key to distinguish where it goes back to
            mover_name=f"tenure_choice_{t+1}_to_owner_consumption_{t}_backward"
        )
        
        # Connect tenure_choice back to renter_consumption in previous period
        model.add_inter_period_connection(
            source_period=model.get_period(t+1),
            target_period=model.get_period(t),
            source_stage=model.get_period(t+1).get_stage("tenure_choice"),
            target_stage=model.get_period(t).get_stage("renter_consumption"),
            source_perch_attr="arvl",
            target_perch_attr="cntn",
            branch_key="to_renter_consumption",  # Branch key to distinguish where it goes back to
            mover_name=f"tenure_choice_{t+1}_to_renter_consumption_{t}_backward"
        )
    
    # Print out movers for Period 0 (intra-period)
    print("\n=== Movers in Period 0 ===")
    period0 = model.get_period(0)
    
    print("Forward Movers (Intra-Period):")
    # Use forward_graph to access forward edges/movers
    for source, target, data in period0.forward_graph.edges(data=True):
        mover = data["mover"]
        # Include more detailed information about perches
        source_perch_attr = data.get("source_perch_attr", "N/A")
        target_perch_attr = data.get("target_perch_attr", "N/A")
        branch_key = mover.branch_key
        print(f"  {mover.name}: {source} -> {target}")
        print(f"     Source perch: {source_perch_attr}, Target perch: {target_perch_attr}, Branch key: {branch_key}")
    
    print("\nBackward Movers (Intra-Period):")
    # Use backward_graph to access backward edges/movers
    for source, target, data in period0.backward_graph.edges(data=True):
        mover = data["mover"]
        source_perch_attr = data.get("source_perch_attr", "N/A")
        target_perch_attr = data.get("target_perch_attr", "N/A")
        branch_key = mover.branch_key
        print(f"  {mover.name}: {source} -> {target}")
        print(f"     Source perch: {source_perch_attr}, Target perch: {target_perch_attr}, Branch key: {branch_key}")
    
    # Print out movers for one inter-period connection
    if num_periods > 1:
        print("\n=== Inter-Period Movers (Period 0 to Period 1) ===")
        print("Forward Movers (Inter-Period):")
        # Find inter-period movers from period 0 to period 1
        for mover in model.inter_period_movers:
            # Check if this mover connects period 0 to period 1
            if hasattr(mover, 'source_period_idx') and hasattr(mover, 'target_period_idx'):
                if (mover.source_period_idx == 0 and 
                    mover.target_period_idx == 1 and
                    mover.edge_type == "forward"):
                    # Get edge data from the model's forward graph
                    edge_data = None
                    for _, _, data in model.forward_graph.edges(data=True):
                        if data.get("mover") == mover:
                            edge_data = data
                            break
                    
                    source_perch_attr = edge_data.get("source_perch_attr", "N/A") if edge_data else "N/A"
                    target_perch_attr = edge_data.get("target_perch_attr", "N/A") if edge_data else "N/A"
                    branch_key = mover.branch_key
                    
                    print(f"  {mover.name}: {mover.source_name} -> {mover.target_name}")
                    print(f"     Source perch: {source_perch_attr}, Target perch: {target_perch_attr}, Branch key: {branch_key}")
        
        print("\nBackward Movers (Inter-Period):")
        for mover in model.inter_period_movers:
            # Check if this mover connects period 1 to period 0
            if hasattr(mover, 'source_period_idx') and hasattr(mover, 'target_period_idx'):
                if (mover.source_period_idx == 1 and 
                    mover.target_period_idx == 0):
                    # Get edge data from the model's backward graph
                    edge_data = None
                    for _, _, data in model.forward_graph.edges(data=True):  # Forward graph because these are manually created forward edges that go backward
                        if data.get("mover") == mover:
                            edge_data = data
                            break
                    
                    source_perch_attr = edge_data.get("source_perch_attr", "N/A") if edge_data else "N/A"
                    target_perch_attr = edge_data.get("target_perch_attr", "N/A") if edge_data else "N/A"
                    branch_key = mover.branch_key
                    
                    print(f"  {mover.name}: {mover.source_name} -> {mover.target_name}")
                    print(f"     Source perch: {source_perch_attr}, Target perch: {target_perch_attr}, Branch key: {branch_key}")
    
    # Create directory for images if it doesn't exist
    image_dir = os.path.join(current_dir, "images", "housing_rental")
    os.makedirs(image_dir, exist_ok=True)
    
    # Visualize the model structure with period-based spring layout
    print("\nVisualizing housing rental model...")
    model.visualize_stage_graph(
        edge_type='both',                 # Show both forward and backward edges
        layout='hierarchical',            # Hierarchical layout instead of spring layout
        node_size=1500,                   # Increased node size
        title="Housing Rental Model - Hierarchical View",
        edge_style_mapping={
            'forward_intra': {'color': '#1f77b4', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
            'backward_intra': {'color': '#d62728', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20},
            'forward_inter': {'color': '#2ca02c', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
            'backward_inter': {'color': '#9467bd', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20}
        },
        show_node_labels=True,
        show_edge_labels=True,
        connectionstyle='arc3,rad=0.1',    # Reduced curve radius for cleaner lines
        filename=os.path.join(image_dir, "housing_rental_model_hierarchical.png")
    )
    
    # Also create a circular layout visualization which can be clearer for this type of model
    model.visualize_stage_graph(
        edge_type='both',                 
        layout='circular',                # Circular layout
        node_size=1500,                   
        title="Housing Rental Model - Circular View",
        edge_style_mapping={
            'forward_intra': {'color': '#1f77b4', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
            'backward_intra': {'color': '#d62728', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20},
            'forward_inter': {'color': '#2ca02c', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
            'backward_inter': {'color': '#9467bd', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20}
        },
        show_node_labels=True,
        show_edge_labels=False,          # No edge labels for cleaner view
        connectionstyle='arc3,rad=0.2',
        filename=os.path.join(image_dir, "housing_rental_model_circular.png")
    )
    
    # Also create a version with just the forward edges for clarity
    model.visualize_stage_graph(
        edge_type='forward',              # Only forward edges
        layout='hierarchical',            
        node_size=1500,                   
        title="Housing Rental Model - Forward Edges Only",
        edge_style_mapping={
            'forward_intra': {'color': '#1f77b4', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
            'forward_inter': {'color': '#2ca02c', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
        },
        show_node_labels=True,
        show_edge_labels=True,
        connectionstyle='arc3,rad=0.1',
        filename=os.path.join(image_dir, "housing_rental_model_forward.png")
    )
    
    # Create an improved spring layout with cleaner presentation
    model.visualize_stage_graph(
        edge_type='both',                 
        layout='period_spring',           # Period-grouped spring layout
        node_size=1800,                   # Larger nodes
        title="Housing Rental Model - Clean Spring Layout",
        edge_style_mapping={
            'forward_intra': {'color': '#1f77b4', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
            'backward_intra': {'color': '#d62728', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20},
            'forward_inter': {'color': '#2ca02c', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
            'backward_inter': {'color': '#9467bd', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20}
        },
        show_node_labels=True,
        show_edge_labels=False,          # No edge labels for cleaner view
        label_offset=0.3,                # Increase label offset from nodes
        connectionstyle='arc3,rad=0.2',  # Curve the edges
        figsize=(14, 10),               # Larger figure size
        dpi=150,                        # Higher resolution
        with_edge_legend=True,          # Add legend for edge types
        filename=os.path.join(image_dir, "housing_rental_model_clean_spring.png")
    )
    
    # Print model structure summary
    print("\nModel Structure Summary:")
    print(f"Model name: {model.name}")
    print(f"Number of periods: {num_periods}")
    print(f"Total number of stages created: {num_periods * 5}")  # 5 stages per period
    print(f"Image saved in: {image_dir}")
    
    return model

if __name__ == "__main__":
    model = main() 