from controller import Supervisor
import PuPy

# checks
checks = []
checks.append(PuPy.RevertTumbled(grace_time_ms=(3 * 3000)))
checks.append(PuPy.RevertMaxIter(3000 * 300))

# set up supervisor
s = PuPy.supervisorBuilder(Supervisor, 20, checks)

# run
s.run()
