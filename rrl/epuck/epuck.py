"""
For the ePuck robot, a small simulator is provided. It allows to place
ePuck in an arena, with unpassable walls and obstacles at (almost)
arbitrary locations. Some environment objects are predefined in
:py:mod:`rrl.epuck.env`. The class :py:class:`Robot` provides the
implementation of the simulated ePuck. Obstacles are directly inserted
into this instance, hence it combines the robot with the environment.
As for other problems, a :py:class:`ADHDP` instance can be used on top
of this to control the robot motion. Both of these parts are combined in
the :py:func:`simulation_loop` function to run the simulation for a
fixed amount of time.

"""
import numpy as np
import pylab
import warnings


def _intersect((o1x, o1y), (d1x, d1y), (o2x, o2y), (d2x, d2y)):
    """Intersection of two bounded lines. The lines are given
    with the origin and direction. Returned is the step length for
    both lines, in the same order as the input.
    
    o1x + t1 * d1x = o2x + t2 * d2x
    o1y + t1 * d1y = o2y + t2 * d2y
    => t1 = (o2x + t2 * d2x - o1x)/d1x
    => o1y + ((o2x + t2 * d2x - o1x)/d1x) * d1y = o2y + t2 * d2y
    => o1y + (o2x + t2 * d2x - o1x) * d1y/d1x = o2y + t2 * d2y
    => o1y + (t2 * d2x + o2x - o1x) * d1y/d1x = o2y + t2 * d2y
    => o1y + t2*d2x*d1y/d1x + (o2x - o1x) * d1y/d1x = o2y + t2 * d2y
    => o1y - o2y + (o2x - o1x) * d1y/d1x = t2 * d2y - t2*d2x*d1y/d1x
    => o1y - o2y + (o2x - o1x) * d1y/d1x = t2 * (d2y - d2x*d1y/d1x)
    => t2 = (o1y - o2y + (o2x - o1x) * d1y/d1x) / (d2y - d2x*d1y/d1x)
    
    """
    tol = 1e-14
    if abs(d1y - 0.0) < tol :
        # o_dir = (!0.0, 0.0)
        if abs(d2y - d2x*d1y/d1x) < tol: # parallel
            t0, t1 = float('inf'), float('inf')
        else:
            nom = o2y - o1y - d1y * (o2x - o1x)/d1x
            denom = (d1y*d2x)/d1x - d2y
            t0 = nom/denom
            t1 = (o2x - o1x + t0 * d2x)/d1x
    else:
        # o_dir = (0.0, !0.0)
        if abs(d2x - d2y*d1x/d1y) < tol: # parallel
            t0, t1 = float('inf'), float('inf')
        else:
            nom = o2x - o1x - d1x * (o2y - o1y)/d1y
            denom = (d1x*d2y)/d1y - d2x
            t0 = nom/denom
            t1 = (o2y - o1y + t0 * d2y) / d1y
    
    return t1, t0

def _in_obstacle(loc, obstacle):
    """Check if a location is within an obstacle.
    
    Assuming the obstacle edges are given in the right order (meaning
    that the polygon is defined through lines between successive
    points).
    
    As reference, the origin is picked. This implies that the obstacle
    must not include the origin.
    
    Edges and corners count as within the obstacle
    
    """
    if any([loc == obs for obs in obstacle]):
        return True
    
    faces = [(p0, p1) for p0, p1 in zip(obstacle[:-1], obstacle[1:])]
    faces.append((obstacle[-1], obstacle[0]))
    
    num_intersect = sum([_obs_intersect((loc, (0.0, 0.0)), line) for line in faces])
    if num_intersect % 2 == 0:
        return False
    else:
        return True

def _obs_intersect(((x0, y0), (x1, y1)), ((x2, y2), (x3, y3))):
    """Check if two lines intersect. The boundaries don't count as
    intersection."""
    base1 = (x0, y0)
    base2 = (x2, y2)
    dir1 = (x1-x0, y1-y0)
    dir2 = (x3-x2, y3-y2)
    t1, t2 = _intersect(base1, dir1, base2, dir2)
    
    eps = 0.00001
    if -eps < t1 and t1 < 1.0 + eps and -eps < t2 and t2 < 1.0 + eps:
        return True
    else:
        return False

class Robot(object):
    """Simulated ePuck robot.
    
    The robot may be steered by means of change in its orientation (i.e.
    the heading relative to the robot). Every time an action is
    executed, the robot turns to the target orientation, then moves
    forward. How much it moves is proportional to the ``speed`` and
    ``step_time``. In between, infrared sensor readouts can be taken.
    The robot is placed in an arena, with some obstacles and walls it
    can collide with but not pass. Upon collision, the robot stops
    moving.
    
    ``walls``
        List of wall lines which cannot be passed. The lines are to be
        given by their endpoints.
    
    ``obstacles``
        List of obstacles which cannot be passed. In contrast to walls,
        the obstacles are closed polygons. They have to be given
        as list of corner points. Obstacles may not include the origin
        (0, 0).
    
    ``speed``
        Speed of the robot.
    
    ``step_time``
        Time quantum for movement, i.e. for how long the robot drives
        forward.
    
    ``tol``
        Minimal distance from any obstacle or wall which counts as
        collision.
    
    .. note::
        Obstacles may not include the origin (0, 0).
    
    .. todo::
        wall tolerance does not operate correctly.
    
    """
    def __init__(self, walls=None, obstacles=None, speed=0.5, step_time=1.0, tol=0.0):
        
        if obstacles is None:
            obstacles = []
        
        if walls is None:
            walls = []
        
        walls = walls[:]
        for obs in obstacles:
            walls.extend([(x0, y0, x1, y1) for (x0, y0), (x1, y1) in zip(obs[:-1], obs[1:])])
            walls.append((obs[-1][0], obs[-1][1], obs[0][0], obs[0][1]))
        
        if tol > 0.0:
            warnings.warn("tolerance > 0 doesn't work properly; It only works if the robot faces the wall (not when parallel or away from the wall).")
        
        self.sensors = [2*np.pi*i/8.0 for i in range(8)]
        #self.obstacles = [ (x0,y0,x1,y1) ]
        self.obstacle_line = walls
        self._ir_max, self.tol = 15.0, tol
        self.obstacles = self._cmp_obstacles(self.obstacle_line)
        self.polygons = obstacles[:]
        self.speed, self.step_time = speed, step_time
        self.loc = (0.0, 0.0)
        self.pose = 0.0
        self.trajectory = []
        self.reset()
    
    def _cmp_obstacles(self, lines):
        """Convert lines given by their endpoints to their corresponding
        vector representation"""
        obstacles = []
        for x0, y0, x1, y1 in lines:
            o_vec = (x1-x0, y1-y0)
            if o_vec[0] == 0.0 and o_vec[1] == 0.0:
                raise Exception('Obstacle line must have a direction')
            o_base = (x0, y0)
            o_limit = 1.0
            obstacles.append((o_vec, o_base, o_limit))
        return obstacles
    
    def _cmp_obstacle_lines(self, obstacles):
        """Convert lines given by as vector to their corresponding
        endpoint representation."""
        lines = []
        for o_vec, o_base, o_limit in obstacles:
            x0, y0 = o_base
            if o_limit == float('inf'):
                raise Exception('Infinite lines not supported')
            x1 = o_base[0] + o_limit * o_vec[0]
            y1 = o_base[1] + o_limit * o_vec[1]
            lines.append((x0, y0, x1, y1))
        return lines
    
    def reset(self):
        """Reset the robot to the origin."""
        self.loc = (0.0, 0.0)
        self.pose = 0.0
        self.trajectory = [self.loc]
    
    def reset_random(self, loc_lo=-10.0, loc_hi=10.0):
        """Reset the robot to a random location, outside the obstacles."""
        for i in xrange(1000):
            loc = self.loc = (np.random.uniform(loc_lo, loc_hi), np.random.uniform(loc_lo, loc_hi))
            pose = self.pose = np.random.uniform(0, 2*np.pi)
            
            if not any([_in_obstacle(self.loc, obs) for obs in self.polygons]) and not self.take_action(0.0):
                break
        
        if i == 1000:
            warnings.warn('Random reset iterations maximum exceeded')
        
        self.loc = loc
        self.pose = pose
        self.trajectory = [self.loc]
    
    def read_ir(self):
        """Compute the proximities to obstacles in all infrared sensor
        directions."""
        # view-direction
        readout = []
        for sensor in self.sensors:
            s_dist = self._ir_max
            s_ori = self.pose + sensor
            s_dir = (np.cos(s_ori), np.sin(s_ori))
            s_base = self.loc
            
            for o_dir, o_base, o_limit in self.obstacles:
                # obstacles intersection
                t0, t1 = _intersect(o_base, o_dir, s_base, s_dir)
                
                eps = 0.00001
                if t1 >= 0 and (o_limit == float('inf') or (-eps <= t0 and t0 <= o_limit + eps)):
                #if t0 >= 0 and t1 >= 0 and t1 <= 1.0:
                    # intersection at distance (t0 * s_dir)
                    dist = np.linalg.norm((t1 * s_dir[0], t1 * s_dir[1]))
                else:
                    # no intersection
                    dist = self._ir_max
                
                if dist < s_dist:
                    s_dist = dist
            
            readout.append(s_dist)
            
        return readout
    
    def read_sensors(self):
        """Read all sensors. A :py:keyword:`dict` is returned."""
        ir = self.read_ir()
        #noise = np.random.normal(scale=0.01, size=(len(ir)))
        #ir = map(operator.add, ir, noise)
        
        return {'loc': np.atleast_2d(self.loc), 'pose': np.atleast_2d(self.pose), 'ir': np.atleast_2d(ir)}
    
    def take_action(self, action):
        """Execute an ``action`` and move forward
        (speed * step_time units or until collision). Return
        :py:const:`True` if the robot collided.
        
        """
        # turn
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]
        self.pose = (self.pose + action) % (2*np.pi)
        #self.pose = action % (2*np.pi)
        
        # move forward
        t = self.speed * self.step_time # distance per step
        
        # Collision detection
        eps = 0.00001
        r_vec = (np.cos(self.pose), np.sin(self.pose))
        wall_dists = [(idx, _intersect(self.loc, r_vec, o_base, o_vec), o_limit) for idx, (o_vec, o_base, o_limit) in enumerate(self.obstacles)]
        wall_dists = [(idx, r_dist) for idx, (r_dist, o_dist, o_limit) in wall_dists if r_dist >= 0.0 and r_dist < float('inf') and -eps <= o_dist and o_dist <= o_limit + eps]
        if len(wall_dists) > 0:
            # Distance to the wall
            wall_idx, min_wall_dist = min(wall_dists, key=lambda (idx, dist): dist)
            dist = np.linalg.norm((min_wall_dist * r_vec[0], min_wall_dist * r_vec[1]))
            
            # angle between wall and robot trajectory
            o_vec = self.obstacles[wall_idx][0]
            a = np.arccos( (o_vec[0] * r_vec[0] + o_vec[1] * r_vec[1]) / (np.linalg.norm(o_vec) * np.linalg.norm(r_vec)) )
            if a > np.pi/2.0:
                a = np.pi - a
            
            # maximum driving distance
            k = self.tol / np.sin(a)
            t_max = dist - k
            
        else:
            # no wall ahead
            t_max = float('inf')
        
        collide = t >= t_max
        t = min(t, t_max)
        
        # next location
        self.loc = (self.loc[0] + np.cos(self.pose) * t, self.loc[1] + np.sin(self.pose) * t) # t doesn't denote the distance in moving direction!
        self.trajectory.append(self.loc)
        return collide
    
    def plot_trajectory(self, wait=False, with_tol=True, tol=None, full_view=True, axis=None):
        """Plot the robot trajectory in a :py:mod:`pylab` figure.
        
        ``wait``
            True for blocking until the figure is closed.
        
        ``with_tol``
            Plot obstacle tolerance lines.
        
        ``tol``
            Overwrite the obstacle tolerance.
        
        ``full_view``
            Keep the original clipping of the window. If false, the
            clipping will be adjusted to the data.
        
        ``axis``
            A :py:mod:`pylab` axis, which should be used for plotting.
            If not provided, the first axis of the first figure is used.
        
        """
        if axis is None:
            axis = pylab.figure(1).axes[0]
            
        axis.clear()
        self._plot_obstacles(axis, with_tol, tol)
        x, y = zip(*self.trajectory)
        axis.plot(x, y, 'b-')
        axis.plot(x, y, 'b*')
        if full_view:
            x0, x1, y0, y1 = axis.axis()
        else:
            x0, x1, y0, y1 = min(x), max(x), min(y), max(y)
        axis.axis((
            x0 + x0*0.1,
            x1 + x1*0.1,
            y0 + y0*0.1,
            y1 + y1*0.1
            ))
        
        pylab.show(block=wait)

    def _plot_obstacles(self, axis, with_tol=True, tol=None):
        """Plot all obstacles and walls into a :py:mod:`pylab` figure.
        
        ``axis``
            The axis where stuff is plotted into.
        
        ``with_tol``
            Plot obstacle tolerance lines.
        
        ``tol``
            Overwrite the obstacle tolerance.
        
        """
        if tol is None:
            tol = self.tol
        
        for vec, base, limit in self.obstacles:
            # obstacle line
            axis.plot((base[0], base[0]+limit*vec[0]), (base[1], base[1]+limit*vec[1]), 'k')
            
            if with_tol and tol > 0:
                if vec[1] == 0.0:
                    y = (-vec[1]/vec[0], 1.0)
                else:
                    y = (1.0, -vec[0]/vec[1])
                    
                y = (y[0] * tol / np.linalg.norm(y), y[1] * tol / np.linalg.norm(y))
                base_tn = (base[0] - y[0], base[1] - y[1])
                base_tp = (base[0] + y[0], base[1] + y[1])
            
                # obstacle tolerance
                axis.plot((base_tn[0], base_tn[0]+limit*vec[0]), (base_tn[1], base_tn[1]+limit*vec[1]), 'k:')
                axis.plot((base_tp[0], base_tp[0]+limit*vec[0]), (base_tp[1], base_tp[1]+limit*vec[1]), 'k:')

class AbsoluteRobot(Robot):
    """Simulated ePuck robot.
    
    In contrast to :py:class:`Robot`, the heading is with respect to
    the arena instead of the robot - i.e. it is absolute, not relative
    to the robot.
    
    """
    def take_action(self, action):
        """Execute an ``action`` and move forward
        (speed * step_time units or until collision). Return
        :py:const:`True` if the robot collided.
        
        """
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]
        self.pose = action % (2*np.pi)
        return super(AbsoluteRobot, self).take_action(0.0)

def simulation_loop(acd, robot, max_step=-1, max_episodes=-1, max_total_iter=-1):
    """Simulate some episodes of the ePuck robot.
    
    This method handles data passing between the ``acd`` and ``robot``
    instances in two loops, one for the episode and one for the whole
    experiment.
    
    ``acd``
        Actor-Critic instance (:py:class:`ADHDP`).
    
    ``robot``
        Robot instance (:py:class:`Robot`).
    
    ``max_step``
        Maximum number of steps in an episode. Negative means no limit.
    
    ``max_episodes``
        Maximum number of episodes. Negative means no limit.
    
    ``max_total_iter``
        Maximum number of steps in total. Negative means no limit.
    
    """
    if max_step < 0 and max_episodes < 0 and max_total_iter < 0:
        raise Exception('The simulation cannot run forever.')
    
    policy = acd.policy
    num_episode = 0
    num_total_iter = 0
    while True:
        
        # init episode
        acd.new_episode()
        robot.reset()
        #robot.reset_random(loc_lo=-9.0, loc_hi=9.0)
        policy.reset()
        a_curr = np.atleast_2d([policy.action])
        
        num_step = 0 # k
        while True:
            
            # Apply current action
            collided = robot.take_action(a_curr)
            
            # Observe sensors
            s_next = robot.read_sensors()
            
            # Execute ACD
            a_next = acd(s_next, num_step, num_step+1, 1)
            
            # Iterate
            num_step += 1
            num_total_iter += 1
            if collided:
                break
            if max_step > 0 and num_step >= max_step:
                break
            acd.a_curr = a_curr = a_next
        
        if num_step <= 3:
            print "Warning: episode ended prematurely"
        
        num_episode += 1
        if max_episodes > 0 and num_episode >= max_episodes:
            break
        if max_total_iter > 0 and num_total_iter >= max_total_iter:
            break
    
    return acd
