
from controller import Supervisor
import PuPy

# checks
checks = [
    PuPy.QuitOnDemand(),
    PuPy.RevertOnDemand()
]

# set up supervisor
s = PuPy.supervisorBuilder(Supervisor, 20, [PuPy.ReceiverCheck(checks)])

# run
s.run()
