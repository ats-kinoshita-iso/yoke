I'm about to describe a new capability. Before writing any implementation code:

1. Write the eval(s) that define success for this capability
2. Run the evals to confirm they fail (establishing the baseline)
3. Only then proceed with implementation
4. After implementation, run evals again to confirm they pass

This is eval-driven development. The eval defines the capability, not the code.
