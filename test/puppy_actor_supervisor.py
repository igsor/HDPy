from controller import Supervisor
import PuPy

# checks
checks = []
checks.append(PuPy.RestartTumbled(grace_time_ms=(3 * 3000)))
checks.append(PuPy.RestartMaxIter(3000 * 300))

# set up supervisor
s = PuPy.supervisorBuilder(Supervisor, 20, checks)

# run
s.run()
