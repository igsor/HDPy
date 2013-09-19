
from controller import Supervisor
import PuPy

# checks
checks = []
checks.append(PuPy.RespawnOutOfArena(arena_size=(-10, 10, -10, 10), reset_policy=0, distance=0))
# respawn the robot at a random location in a bounded area
checks.append(PuPy.RespawnTumbled(arena_size=(-5, 5, -5, 5), reset_policy=2, grace_time_ms=(3 * 3000)))

# set up supervisor
s = PuPy.supervisorBuilder(Supervisor, 20, checks)

# run
s.run()
