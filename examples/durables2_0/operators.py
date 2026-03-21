"""Stage operator factories for the durables model.

keeper_cons:  EGM + FUES (horses/keeper_egm.py)
adjuster_cons: partial EGM + FUES (Operator_Factory)
tenure:       transitions + eval + max + E_z (horses/branching.py)
"""

import numpy as np
from examples.durables.durables import Operator_Factory
from .horses.keeper_egm import make_keeper_ops
from .horses.branching import make_tenure_ops


def build_stage_ops(model):
    """Build all three stage operators."""
    cp = model.cp
    (_, _, condition_V,
     condition_V_HD, _, internals) = Operator_Factory(cp)

    _adjEGM = internals['_adjEGM']
    _refine_adj = internals['refine_adj']
    return_grids = cp.return_grids

    # Keeper (self-contained horse)
    _keeper_dcsn = make_keeper_ops(model)

    # Tenure (self-contained horse)
    _tenure_dcsn, _tenure_arvl, _tenure_arvl_hd = \
        make_tenure_ops(
            model, condition_V, condition_V_HD)

    # --- adjuster_cons (still from Operator_Factory) ---

    du_c_op = cp.du_c

    def adjuster_dcsn_mover(vlu_cntn, t, m_bar=1.4):
        """Adjuster partial EGM + root-finding + FUES.

        Returns policies + dV_w (marginal value of wealth).
        """
        egrid, vf, a_nxt, h_nxt = _adjEGM(
            vlu_cntn['dV']['a'], vlu_cntn['dV']['h'],
            vlu_cntn['V'], t)
        (Aadj, Cadj, Hadj, Vadj,
         _, _, _, _, _, _, _, _) = _refine_adj(
            egrid, vf, a_nxt, h_nxt,
            m_bar=m_bar, return_grids=return_grids)
        # dV_w = du_c(c) for each z on wealth grid
        n_z_loc = Cadj.shape[0]
        n_w = Cadj.shape[1]
        dVw_adj = np.empty((n_z_loc, n_w))
        for iz in range(n_z_loc):
            for iw in range(n_w):
                dVw_adj[iz, iw] = du_c_op(Cadj[iz, iw])
        return Aadj, Cadj, Hadj, Vadj, dVw_adj

    return {
        'keeper_cons': {
            'dcsn_mover': _keeper_dcsn,
        },
        'adjuster_cons': {
            'dcsn_mover': adjuster_dcsn_mover,
        },
        'tenure': {
            'dcsn_mover': _tenure_dcsn,
            'arvl_mover': _tenure_arvl,
            'arvl_mover_hd': _tenure_arvl_hd,
        },
    }
