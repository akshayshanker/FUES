"""Stage operator factories for the durables model.

Three stages matching the YAML period template:

- ``keeper_cons``: keeper EGM + FUES + interp to state grid
- ``adjuster_cons``: partial EGM + FUES + Bellman eval on state grid
- ``tenure``: branching max + E_z

Each factory returns ``{'dcsn_mover', 'arvl_mover'}``.

Phase 2: leaf arvl_movers interpolate onto the common
(z, a, h) grid. The branching stage receives only
branch-keyed results — no continuation values.
"""

import numpy as np
from examples.durables.durables import Operator_Factory
from .horses.keeper_egm import make_keeper_ops


def build_stage_ops(model):
    """Build all three stage operators.

    Keeper: from horses/keeper_egm.py (make_egm_1d + FUES).
    Adjuster + tenure: from Operator_Factory internals.

    Parameters
    ----------
    model : DurablesModel

    Returns
    -------
    dict
        Stage operators keyed by YAML stage name.
    """
    cp = model.cp
    (_, _, condition_V,
     condition_V_HD, _, internals) = Operator_Factory(cp)

    _adjEGM = internals['_adjEGM']
    _refine_adj = internals['refine_adj']
    _keeper_to_grid = internals['_keeper_to_state_grid']
    _adjuster_to_grid = internals['_adjuster_to_state_grid']
    _branching_max = internals['_branching_max']

    return_grids = cp.return_grids
    delta = cp.delta

    # --- keeper_cons (from horse) ---

    _keeper_dcsn = make_keeper_ops(model)

    # keeper_dcsn_mover is _keeper_dcsn directly.
    # Caller (solve_period) passes h_keep_grid from
    # the tenure branch transition.

    def keeper_arvl_mover(Akeeper, Ckeeper, Vkeeper,
                          w_keep, h_keep, vlu_cntn, t):
        """I: interp keeper onto state grid via w_keep.

        w_keep and h_keep come from tenure branch_transitions.
        Interpolates keeper policies (on asset grid) to the
        state grid using w_keep[z, a] = R*a + y(t,z).
        Also computes phi (housing marginal).
        """
        v_sg, c_sg, a_sg, h_sg, phi_sg = \
            _keeper_to_grid(
                t, Akeeper, Ckeeper, Vkeeper,
                vlu_cntn['dV']['h'],
                vlu_cntn['dV'].get('h_hd',
                    np.zeros((1, 1, 1))))
        return (
            {'c': c_sg, 'a_nxt': a_sg, 'h_nxt': h_sg},
            {'V': v_sg, 'phi': phi_sg},
        )

    # --- adjuster_cons ---

    def adjuster_dcsn_mover(vlu_cntn, t, m_bar=1.4):
        """B: partial EGM + root-finding + FUES.

        Returns refined policies on the wealth grid.
        """
        egrid, vf, a_nxt, h_nxt = _adjEGM(
            vlu_cntn['dV']['a'], vlu_cntn['dV']['h'],
            vlu_cntn['V'], t)
        (Aadj, Cadj, Hadj, Vadj,
         _, _, _, _, _, _, _, _) = _refine_adj(
            egrid, vf, a_nxt, h_nxt,
            m_bar=m_bar, return_grids=return_grids)
        return Aadj, Cadj, Hadj, Vadj

    def adjuster_arvl_mover(Aadj, Cadj, Hadj, Vadj,
                            w_adj, vlu_cntn, t,
                            c_from_budget=1):
        """I: interp adjuster onto state grid via w_adj.

        w_adj comes from tenure branch_transitions.
        """
        v_sg, c_sg, a_sg, h_sg = _adjuster_to_grid(
            t, Aadj, Cadj, Hadj,
            vlu_cntn['V'], c_from_budget)
        return (
            {'c': c_sg, 'a_nxt': a_sg, 'h_nxt': h_sg},
            {'V': v_sg},
        )

    # --- tenure ---

    R = cp.R
    R_H = cp.R_H
    y_func = cp.y_func

    def tenure_arvl_to_dcsn(t):
        """Tenure arvl_to_dcsn: compute branch grids.

        Per YAML dcsn_to_cntn_transition:
          keep:   w_keep = R*a + y(z), h_keep = (1-delta)*h
          adjust: w_adj = R*a + R_H*(1-delta)*h + y(z)

        Returns wealth/housing grids for each branch.
        """
        z = cp.z_vals
        a = cp.asset_grid_A
        h = cp.asset_grid_H
        n_z, n_a, n_h = len(z), len(a), len(h)

        w_keep = np.empty((n_z, n_a))
        for iz in range(n_z):
            w_keep[iz, :] = R * a + y_func(t, z[iz])

        w_adj = np.empty((n_z, n_a, n_h))
        for iz in range(n_z):
            for ih in range(n_h):
                w_adj[iz, :, ih] = (
                    R * a + R_H * (1 - delta) * h[ih]
                    + y_func(t, z[iz]))

        h_keep = (1 - delta) * h

        return {
            'w_keep': w_keep,
            'h_keep': h_keep,
            'w_adj': w_adj,
        }

    def decision_dcsn_mover(t, branches):
        """B: max over keep/adjust branches.

        Clean DDSL interface: receives only branch-keyed
        results from the leaf stages. No vlu_cntn.

        Parameters
        ----------
        t : int
        branches : dict
            ``{'keep':   {'pol': {...}, 'vlu': {...}},
               'adjust': {'pol': {...}, 'vlu': {...}}}``

        Returns
        -------
        vlu_dcsn : dict
        pol : dict
        """
        keep = branches['keep']
        adj = branches['adjust']

        V, A, H, C, D, dV_a, dV_h = _branching_max(
            keep['vlu']['V'],
            keep['pol']['c'],
            keep['pol']['a_nxt'],
            keep['pol']['h_nxt'],
            keep['vlu']['phi'],
            adj['vlu']['V'],
            adj['pol']['c'],
            adj['pol']['a_nxt'],
            adj['pol']['h_nxt'],
        )

        vlu_dcsn = {'V': V, 'dV': {'a': dV_a, 'h': dV_h}}
        pol = {'c': C, 'a_nxt': A, 'h_nxt': H, 'd': D}
        return vlu_dcsn, pol

    def decision_arvl_mover(vlu_dcsn):
        """I: E_z conditioning (Pi @ arrays)."""
        Ev, Edv_a, Edv_h = condition_V(
            vlu_dcsn['V'],
            vlu_dcsn['dV']['a'],
            vlu_dcsn['dV']['h'])
        return {'V': Ev, 'dV': {'a': Edv_a, 'h': Edv_h}}

    def decision_arvl_mover_hd(dV_h_hd):
        """I: E_z conditioning for HD grid."""
        return condition_V_HD(dV_h_hd)

    return {
        'keeper_cons': {
            'dcsn_mover': _keeper_dcsn,
            'arvl_mover': keeper_arvl_mover,
        },
        'adjuster_cons': {
            'dcsn_mover': adjuster_dcsn_mover,
            'arvl_mover': adjuster_arvl_mover,
        },
        'tenure': {
            'arvl_to_dcsn': tenure_arvl_to_dcsn,
            'dcsn_mover': decision_dcsn_mover,
            'arvl_mover': decision_arvl_mover,
            'arvl_mover_hd': decision_arvl_mover_hd,
        },
    }
