# FUES ue_kwargs Debugging Session - 30 Nov 2025

## Problem
FUES method-specific kwargs (`left_turn_no_jump_strict`, `use_post_state_jump_test`, etc.) defined in `master.yml` were not being passed to the `_scan` function.

## Root Cause Discovery

### Initial Symptoms
- Debug output showed `ue_kwargs={}` (empty dict)
- Changed values in `master.yml` had no effect on solver behavior

### Investigation Path

1. **Traced the call chain**: `horses_c.py` → `EGM_UE` → `_fues_engine` → `FUES()` → `_scan()`
   - All connections were correct

2. **Added debug prints** to trace values at each step

3. **Discovered the issue**: `model.settings_dict` was getting `ue_kwargs` as an empty dict

4. **Found root cause in dynx**: 
   - `settings_dict` is populated from **stage configs** (OWNC.yml, RNTC.yml), not master.yml
   - Stage config had `ue_kwargs: ["ue_kwargs"]` reference
   - The reference resolution **does work** through `resolve_parameter_references()` in dynx

5. **The actual bug**: We were testing with an **old cached Numba version** of the code
   - Numba caches compiled `@njit` functions
   - Even after code changes, old compiled code was being used

## Solution

1. **Always clear Numba cache** at job start:
   ```bash
   export NUMBA_CACHE_DIR=/scratch/tp66/$USER/numba_cache
   rm -rf "$NUMBA_CACHE_DIR"
   mkdir -p "$NUMBA_CACHE_DIR"
   ```

2. **Reference resolution works**: Stage configs can use `["setting_name"]` to pull from master.yml:
   ```yaml
   # In stage config (OWNC.yml):
   settings:
     ue_kwargs: ["ue_kwargs"]  # Resolves from master.yml
   ```

3. **Master config structure**:
   ```yaml
   # In master.yml:
   settings:
     ue_kwargs:
       FUES:
         m_bar: 1.0
         lb: 4
         left_turn_no_jump_strict: true
         use_post_state_jump_test: true
         # ... other FUES-specific kwargs
       DCEGM: {}
       CONSAV: {}
   ```

## Key Lessons

1. **Numba caching is aggressive** - Always clear cache when debugging @njit function behavior
2. **dynx reference resolution works** - `["key"]` syntax in stage configs resolves from master
3. **Add debug prints at boundaries** - Print values where data crosses module boundaries
4. **Trace the full chain** - Config → model → solver → algorithm

## New FUES Parameters Added

- `left_turn_no_jump_strict`: If True, left turns without jumps use same logic as left turns with jumps
- `use_post_state_jump_test`: If True, jump detection uses post-state gradient (g_tilde_a_2) in addition to pre-state

## Files Modified

- `src/dc_smm/fues/fues.py`: Added new kwargs to `_scan()` and `FUES()`
- `src/dc_smm/uenvelope/upperenvelope.py`: Forward kwargs through `_fues_engine`
- `examples/housing_renting/config_HR/*/master.yml`: Added full FUES kwargs
- `examples/housing_renting/config_HR/*/stages/*.yml`: Added `ue_kwargs` reference
- `experiments/housing_renting/*.pbs`: Added Numba cache clearing

## Verification

Debug output confirming kwargs are passed:
```
[DEBUG horses_c] ue_kwargs raw: {'FUES': {'m_bar': 1.00001, 'lb': 10, ...
  'use_post_state_jump_test': True, 'left_turn_no_jump_strict': True, ...}}
[FUES DEBUG] left_turn_no_jump_strict=True, use_post_state_jump_test=True
```

