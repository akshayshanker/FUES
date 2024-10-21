# Fast Upper-Envelope Scan (FUES) for Discrete-Continuous Dynamic Programming

Repository contains FUES module and examples fast upper-envelope scan (FUES) to clean sub-optimal local optima generated by endogenous grid methods (EGM) in discrete-continuous dynamic programming models.

See `slides....pdf` for overview of upper envelope scan. 

## Example use of FUES

Suppose we have the following arrays: an unrefined endogenous grid `x_hat`, the value correspondence on the unrefined grid `v_hat` 
and two policy functions, `policy_1` and `policy_2_hat`.

```
from FUES.FUES import FUES

x_clean, vf_clean, policy_1_clean, policy_2_clean, dela_clean \
        = FUES(x_hat, v_hat, policy_1_hat, policy_2_hat, dela, LB = 10, m_bar = 0.1, endog_mbar = True)
```

In the above:
- `x_hat` is the unrefined endogenous grid
- `v_hat` is the value correspondence on the unrefined grid
- `policy_1_hat` is the first policy function
- `policy_2_hat` is the second policy function
- `dela` is the derivative of the second policy function
- `LB` is the number of steps to take in the forward and backward scans
- `m_bar` is the maximum possible gradient of the policy function
- `endog_mbar` is a boolean indicating whether the cut off gradient used by FUES is endogenously determined using dela

The derivative of time t policy function can be calculated using the implicit function theorem and the Euler equation (see Section 2.1.4 of the latest working paper).  

It is also possible to set endog_mbar = False and set m_bar = \bar{L} where \bar{L} is the maximum possible gradient of the policy function. 

When m_bar is False, dela is not used and the cut off gradient is set to m_bar (pass in a dummy array for dela).

Note: FUES detects jumps in the the second policy, policy_2_hat. 

## Application 1 (Retirement Choice)

Run the baseline timings, performance and plots. 

```
python3 scripts/retirement_plot.py
```

Table of results:

| Method | Euler Error    | Avg. upper env. time (ms) |
|--------|----------------|---------------------------|
| RFC    | -1.535907      | 11.572099                 |
| FUES   | -1.535912      | 0.813305                  |
| DCEGM  | -1.535847      | 5.644548                  |


Run full timings across grid sizes and delta, set `run_performance_tests = True` in main block of `retirement_plot.py`. 

Table of performance saved in `results/retirement_timings.tex`.

### Plots 


Consumption policy function generated using Ishkakov et al (2017) params and no smoothing:

![ret_cons_all](https://user-images.githubusercontent.com/8477783/181183127-4bf48f5b-8280-4f9f-afe1-1730894c0e29.png)

Upper envelope generation using FUES and Ishkakov et al (2017) params (age 17):

![ret_vf_aprime_all_17](/results/plots/retirement/ret_vf_aprime_all_17_3000_sigma0.png)

Upper envelope and policy functions for Ishkakov et al (2017) params and smoothing param = 0.05:

![ret_vf_aprime_all_17_sigma](https://user-images.githubusercontent.com/8477783/181172404-1b0bbb74-5c40-47c0-aff9-0d34b573f7f2.png)

![ret_cons_all_sigma](https://user-images.githubusercontent.com/8477783/181172415-72f866b9-348e-4de9-9855-fb509591deb2.png)

Comparison with DC-EGM (age 17):

![ret_vf_aprime_all_2000_cf_17](https://user-images.githubusercontent.com/8477783/216878773-3d031849-c26d-46a3-a231-7b19f1a8d793.png)

