"""Package for analysis of STARFORGE simulations."""


# star_gas_columns pulls in numpy + scipy.spatial (~2 s cold on GPFS), so load it
# lazily: console scripts like plot_sf import this package but never touch scipy.
def __getattr__(name):
    if name == "star_gas_columns":
        from .star_gas_columns import star_gas_columns
        return star_gas_columns
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
