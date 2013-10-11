"""
"""
import HDPy
import numpy as np
import pylab

## PARAMS ##

washout         = 200
num_train       = 5000
num_test        = 1000
reservoir_size  = 100

## INITIALIZATION ##

# Reservoir
reservoir_sparse = HDPy.ReservoirNode(
    input_dim       = 1,
    output_dim      = reservoir_size,
    spectral_radius = 0.9,
    w_bias          = None,
    w               = HDPy.sparse_reservoir(20),
    w_in            = HDPy.sparse_w_in(1.5, 50, rnd_fu=np.random.normal),
)

reservoir_orthogonal = HDPy.ReservoirNode(
    input_dim       = 1,
    output_dim      = reservoir_size,
    spectral_radius = 0.9,
    w_bias          = None,
    w               = HDPy.orthogonal_reservoir(20.0),
    w_in            = HDPy.sparse_w_in(1.5, 50, rnd_fu=np.random.normal),
)

reservoir_ring = HDPy.ReservoirNode(
    input_dim       = 1,
    output_dim      = reservoir_size,
    spectral_radius = 0.9,
    w_bias          = None,
    w               = HDPy.ring_of_neurons,
    w_in            = HDPy.sparse_w_in(1.5, 50, rnd_fu=np.random.normal),
)


# Readout
readout = HDPy.StabilizedRLS(
    input_dim       = reservoir_size,
    output_dim      = 1,
    with_bias       = True,
    lambda_         = 1.0,
)

readout_orthogonal  = readout.copy()
readout_ring        = readout.copy()
readout_sparse      = readout.copy()

# Data
def narma30(num_samples=1000):
    """30th order NARMA dataset. Copied from [Oger]_."""
    system_order = 30
    inputs = np.random.rand(num_samples, 1) * 0.5
    outputs = np.zeros((num_samples, 1))
    
    for k in range(system_order-1, num_samples-1):
        outputs[k + 1] = 0.2 * outputs[k] + 0.04 * \
        outputs[k] * np.sum(outputs[k - (system_order-1):k+1]) + \
        1.5 * inputs[k - 29] * inputs[k] + 0.001
    return inputs, outputs 

src, trg = narma30(washout + num_train + num_test)

## TRAINING ##

setups = ('Sparse', 'Orthogonal', 'Ring of Neurons')
reservoirs = (reservoir_sparse, reservoir_orthogonal, reservoir_ring)
readouts = (readout_sparse, readout_orthogonal, readout_ring)

# Initialize the reservoirs
# Propagate data through the reservoirs, no training
for res in reservoirs:
    res(src[:washout])

# Train the readout
# Propagate data through reservoir, train the readout online
for res, out in zip(reservoirs, readouts):
    r_state = res(src[washout:num_train])
    out.train(r_state, trg[washout:num_train])

# Test the networks
signals = []
for res, out in zip(reservoirs, readouts):
    r_state = res(src[washout+num_train:])
    pred = out(r_state)
    signals.append(pred)

## PLOTTING ##

# Error measurement
mse     = lambda sig_pred, sig_trg: ((sig_pred - sig_trg)**2).mean()
rmse    = lambda sig_pred, sig_trg: np.sqrt(mse(sig_pred, sig_trg))
nrmse   = lambda sig_pred, sig_trg: rmse(sig_pred, sig_trg) / sig_trg.std()

# Output and reservoir output plotting
pretty_str = "{0:<" + str(max(map(len, setups))) + "}\t{1:0.6f}\t{2:0.6f}"
print "Reservoir type\tMSE\t\tNRMSE"
for sig, lbl in zip(signals, setups):
    pylab.plot(sig, label=lbl)
    err_mse = mse(sig, trg[washout + num_train:])
    err_nrmse = nrmse(sig, trg[washout + num_train:])
    print pretty_str.format(lbl, err_mse, err_nrmse)

# Target plotting
pylab.plot(trg[washout+num_train:], 'c', label='Target')

# Show the plot
pylab.axis((0.0, 70.0, 0.0, 0.45))
pylab.legend(loc=0)
pylab.show(block=False)
