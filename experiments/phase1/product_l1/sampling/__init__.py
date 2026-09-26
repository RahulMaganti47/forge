"""Drawing products from a trained model: end-to-end and joint samplers, decoders, terminal
generation and the census of what came out. Execution, as distinct from the architecture in
`flow` and the fitting in `training`.


`rstar` holds the Euler step every flow in `design` calls. It was a top-level
`forge.generate` package holding one module that only this package used -- the same
one-file category folder that `eval/` and `verify/` were, so it now sits beside its callers.
"""
