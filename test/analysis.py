
import rrl,pylab
a = rrl.Analysis('esn_acd.hdf5')

# Prediction plot
fig = pylab.figure(1)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
a.plot_path_return_prediction('18', ax)
fig.suptitle('Predictor evaluation')
# If all works well, you should see a figure with 5 lines in it,
# according to the Analysis.plot_path_return_prediction documentation

# Simple plot functions
fig = pylab.figure(2)
a.plot_readout_sum(fig.add_subplot(321))
a.plot_reward(fig.add_subplot(322))
a.plot_derivative(fig.add_subplot(323))
a.plot_actions(fig.add_subplot(324))
a.plot_error(fig.add_subplot(325))
a.plot_accumulated_reward(fig.add_subplot(326))
fig.suptitle('Some characteristics')
# If all works well, you should see 6 subplots with the respective
# curves displayed.

# Show the plot
pylab.show(block=False)

print "Check the graphs visually. If they correspond to your expectations, the test was successful."
