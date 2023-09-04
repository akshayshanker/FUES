# FUES_EGM
The Endogenous grid method (EGM) using fast upper-envelope scan (FUES).

Initial beta replication material for `Fast upper-envelope scan for discrete-continuous dynamic programming' by Dobrescu and Shanker (2023). 

See `slides....pdf` for overview of upper envelope scan. 

## Example use of FUES

Suppose we have the following arrays: an unrefined endogenous grid `x_hat`, the value correspondence on the unrefined grid `v_hat` 
and two policy functions, `c_hat` and `a_prime_hat`.

```
from FUES.FUES import FUES

x_clean, vf_clean, c_clean, a_prime_clean, dela \
        = FUES(x_hat, v_hat, c_hat, a_prime_hat, dela, LB = 10)
```

The outputs are the refined grids, `dela` is the time t policy function derivative used to 
endogenously detect jumps in the policy function and `LB` is the number of steps to take in the forward and backward scans. 

The derivative of time t policy function can be derived using the implicit function theorem and the Euler equation. The derivative can then be evaluated at the exogenous grid points (see Section 2.1.4 of the latest working paper).  

## Application 1

### Plots 


Consumption policy function generated using Ishkakov et al (2017) params and no smoothing:

![ret_cons_all](https://user-images.githubusercontent.com/8477783/181183127-4bf48f5b-8280-4f9f-afe1-1730894c0e29.png)

Upper envelope generation using FUES and Ishkakov et al (2017) params (age 17):

![ret_vf_aprime_all_17](https://user-images.githubusercontent.com/8477783/216878574-7d240142-8e47-49e4-a0d6-98b6f460710c.png)

Upper envelope and policy functions for Ishkakov et al (2017) params and smoothing param = 0.05:

![ret_vf_aprime_all_17_sigma](https://user-images.githubusercontent.com/8477783/181172404-1b0bbb74-5c40-47c0-aff9-0d34b573f7f2.png)

![ret_cons_all_sigma](https://user-images.githubusercontent.com/8477783/181172415-72f866b9-348e-4de9-9855-fb509591deb2.png)

Comparison with DC-EGM (age 17):

![ret_vf_aprime_all_2000_cf_17](https://user-images.githubusercontent.com/8477783/216878773-3d031849-c26d-46a3-a231-7b19f1a8d793.png)

### Comparison to DC-EGM

The following code block in `retirement_plot.py` compares DC-EGM with FUES across an array of parameter values:

https://github.com/akshayshanker/FUES_EGM/blob/fa725a6285fee9def7e8ad07606603a658cdccf6/retirement_plot.py#L381-L470

To perform the comparison, we first solve the full model using FUES, which gives the final solution computed using FUES 
and also the unrefined endogenous grids for each age. For a given age, we then compute the upper envelope using DC-EGM
and FUES. The upper envelopes are compared on the optimal endogenous grid points as determined by DC-EGM. 

(Compared on optimal points to avoid picking up errors arising from different interpolation steps used 
by DC-EGM and FUES. DC-EGM interpolates line  segments on the unrefined grid while FUES first calculates the optimal points then 
interpolates over the unrefined grid.)

```
Test DC-EGM vs. FUES on uniform grid of 1000 parameters:
 beta: (0.85,0.98), y: (10,25), delta: (0.5,1.5)
Avg. error between DC-EGM and FUES: 0.000000
Timings:
 Avg. FUES time (secs): 0.001375
 Avg. worker iteration time (secs): 0.004314
 ```
