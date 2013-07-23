
import rrl
import PuPy
import pylab
import numpy as np

# Create and initialize Policy
gait = PuPy.Gait(params={
    'frequency' : (1.0, 1.0, 1.0, 1.0),
    'offset'    : ( -0.23, -0.23, -0.37, -0.37),
    'amplitude' : ( 0.56, 0.56, 0.65, 0.65),
    'phase'     : (0.0, 0.0, 0.5, 0.5)
})

policy = rrl.FRA(gait)

# Plot action
it = policy.get_iterator(0, 100, 20)
pylab.subplot(311)
data = [it.next() for i in range(100)]
pylab.title('Motor action, untampered for 100 steps, 20ms each')
pylab.xlabel('time')
pylab.plot(data)
pylab.show(block=False)


# Create and initialize Plant
plant = rrl.SpeedReward()

# Create and initialize ACD
reservoir = rrl.SparseReservoirNode(
    output_dim=10,
    input_dim=policy.action_space_dim() + plant.state_space_dim(),
    reset_states=False,
    spectral_radius=0.9,
    fan_in_i=100,
    fan_in_w=20
)

readout = rrl.StabilizedRLS(
    with_bias=True,
    input_dim=10,
    output_dim=1,
    lambda_=1.0
)

expfile = '/tmp/acd.hdf5'
acd = rrl.CollectingADHDP(#  rrl.ADHDP(
    expfile,
    reservoir,
    readout,
    plant,
    policy
)

def my_alpha(time):
    return 0.5

acd.set_alpha(my_alpha)

ep0 = {
    'accelerometer_z'   : np.array([2.0]),
    'puppyGPS_x'        : np.array([0.0,  1.0]),
    'puppyGPS_y'        : np.array([0.0, 10.0])
    }

ep1 = {
    'accelerometer_z'   : np.array([2.0]),
    'puppyGPS_x'        : np.array([ 1.0,  3.0]),
    'puppyGPS_y'        : np.array([10.0, 18.0])
    }

# Initialize for some epochs
it = acd(ep0, time_start_ms=  0, time_end_ms=100, step_size_ms=20)
it = acd(ep0, time_start_ms=100, time_end_ms=200, step_size_ms=20)
it = acd(ep0, time_start_ms=200, time_end_ms=300, step_size_ms=20)

# First epoch
it = acd(ep0, time_start_ms=300, time_end_ms=400, step_size_ms=20)
data = [it.next() for i in range(100)]
pylab.subplot(312)
pylab.title('')
pylab.xlabel('time')
pylab.plot(data)
pylab.show(block=False)

# Second epoch
it = acd(ep1, time_start_ms=400, time_end_ms=500, step_size_ms=20)
data = [it.next() for i in range(100)]
pylab.subplot(313)
pylab.title('')
pylab.xlabel('time')
pylab.plot(data)
pylab.show(block=False)

# Test load/save
import tempfile, os
fh, pth = tempfile.mkstemp()
acd.save(pth)
acd2 = rrl.ADHDP.load(pth)
os.unlink(pth)
