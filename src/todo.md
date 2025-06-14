- fix circular definitions causing overflow
- fix chained comparison
- generate runtime errors (e.g., "ranges must be arithmetic sequences") locally
  and gracefully without halting everything 
- register based vm? maybe easier to codegen for since you don't have to
  maintain the stack's invariants
