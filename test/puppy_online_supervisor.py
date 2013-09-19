
from controller import Supervisor
import PuPy

## REVERT (Const init)
checks = []
checks.append(PuPy.RevertTumbled(grace_time_ms=(3 * 3000)))
checks.append(PuPy.RevertOutOfArena(arena_size=(-10, 10, -10, 10), distance=0))

# set up supervisor
s = PuPy.supervisorBuilder(Supervisor, 20, checks)

# run
s.run()
