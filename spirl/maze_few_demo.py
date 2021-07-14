import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
from spirl.utils.pickle import load

NOISY = True
TARGET_LOCATIONS = {'lr': np.array([7, 1]),
                    'll': np.array([1, 1]),
                    'ul': np.array([1, 10]),
                    'mr': np.array([5, 4]),
                    'left': np.array([3, 10]),
                    'right': np.array([7, 10])}


def get_demo(minimum_length, num_demos, target='lr'):
    env = gym.make('maze2d-large-v1')
    maze = env.str_maze_spec

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)

    env.set_target(TARGET_LOCATIONS[target] + env.np_random.uniform(low=-.1, high=.1, size=env.model.nq))

    all_states = []
    all_actions = []

    i = 0
    while i < num_demos:
        s = env.reset()
        # if constraint == 'right':
        #     while s[0] < 3.5:
        #         s = env.reset()
        # elif constraint == 'left':
        #     while s[0] >= 3.5:
        #         s = env.reset()

        states = [s.copy()]
        actions = []
        while True:
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, env._target)
            if NOISY:
                act = act + np.random.randn(*act.shape) * 0.5

            act = np.clip(act, -1.0, 1.0)

            ns, _, _, _ = env.step(act)

            states.append(ns.copy())
            actions.append(act.copy())

            if done:
                controller = waypoint_controller.WaypointController(maze)
                done = False
                break
            else:
                s = ns
        if len(actions) >= minimum_length:
            all_states.append(states)
            all_actions.append(actions)
            i += 1

    return all_states, all_actions


def process_demo(all_states, all_actions, H):
    states_pool = []
    actions_pool = []
    for i in range(len(all_states)):
        for j in range(len(all_actions[i]) - H + 1):
            states_pool.append(all_states[i][j:j+H+1])
            actions_pool.append(all_actions[i][j:j+H])
    return np.array(states_pool), np.array(actions_pool)


def get_demo_from_file(path, num_demo=10):
    demos = load(path)
    return demos['states'][:num_demo], demos['actions'][:num_demo]
