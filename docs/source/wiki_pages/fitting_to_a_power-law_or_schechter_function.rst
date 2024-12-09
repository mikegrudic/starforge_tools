Fitting data to a power-law or Schechter function
-------------------------------------------------

A common task in astronomy is determining the power-law or Schechter function that a given dataset is most likely to have been drawn from. The powerlaw package makes this straightforward.

::

   import powerlaw
   data = np.loadtxt("IMF_M2e4.dat")[:,1] # stellar mass data from STARFORGE simulation
   results = powerlaw.Fit(data[data>1]) # fitting just the >1msun tail

   slope, mstar = results.truncated_power_law.parameter1, -1/results.truncated_power_law.parameter2 # slope and Schechter cutoff, where the distribution is m^slope exp(-m/mstar)