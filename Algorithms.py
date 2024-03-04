import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


class BFSAgent():

    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        curr = env.get_initial_state()

        if env.is_final_state(curr):
            return [], 0, 0

        open = [curr]
        close = []
        fathers = {curr: (None, -1)}
        costs = {curr: 0}
        terminated = {curr: False}

        while not len(open) == 0:
            curr = open[0]
            open = open[1:]
            close.append(curr)

            if (terminated[curr]):
                continue

            for action, ((position, _, _), cost, ter) in env.succ(curr).items():
                state = (position, curr[1] or (position == env.d1[0]), curr[2] or (position == env.d2[0]))

                if (state not in close) and (state not in open):
                    if env.is_final_state(state):
                        # Calculate track to the solution
                        actions = [action]
                        iterator = curr
                        while fathers[iterator][1] != -1:
                            actions = [fathers[iterator][1]] + actions
                            iterator = fathers[iterator][0]

                        return actions, costs[curr] + cost, len(close)

                    open.append(state)
                    terminated[state] = ter
                    fathers[state] = (curr, action)
                    costs[state] = costs[curr] + cost

        return [], 0, 0


class WeightedAStarAgent():

    def h_msap(self, env: DragonBallEnv, s):
        def h_manheten(state1, state2):
            (x1, y1) = env.to_row_col(state1)
            (x2, y2) = env.to_row_col(state2)

            return np.abs(x1 - x2) + np.abs(y1 - y2)

        min = np.inf
        if (h_manheten(s, env.d1) < min and not s[1]):
            min = h_manheten(s, env.d1)

        if (h_manheten(s, env.d2) < min and not s[2]):
            min = h_manheten(s, env.d2)

        for state in env.get_goal_states():
            if (h_manheten(s, state) < min):
                min = h_manheten(s, state)

        return min

    def f(self, env, state, g, w):
        return (1 - w) * g + w * self.h_msap(env, state)

    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        curr = env.get_initial_state()

        if env.is_final_state(curr):
            return [], 0, 0

        open = heapdict.heapdict()
        close = []
        fathers = {curr: (None, -1)}
        gvalues = {curr: 0}
        terminated = {curr: False}
        expanded = 0

        fvalues = {}
        fvalues[curr] = self.f(env, curr, gvalues[curr], h_weight)
        open[curr] = (self.f(env, curr, gvalues[curr], h_weight), curr[0])

        while not len(open) == 0:
            curr = open.popitem()[0]

            if env.is_final_state(curr):
                # Calculate track to the solution
                actions = []
                iterator = curr
                while fathers[iterator][1] != -1:
                    actions = [fathers[iterator][1]] + actions
                    iterator = fathers[iterator][0]

                return actions, gvalues[curr], expanded

            expanded += 1
            close.append(curr)

            if terminated[curr]:
                continue

            for action, ((position, _, _), cost, ter) in env.succ(curr).items():
                state = (position, curr[1] or (position == env.d1[0]), curr[2] or (position == env.d2[0]))

                new_g = gvalues[curr] + cost
                new_f = self.f(env, state, new_g, h_weight)

                if (state not in close) and (state not in open):
                    fathers[state] = (curr, action)
                    gvalues[state] = new_g
                    fvalues[state] = new_f
                    terminated[state] = ter
                    open[state] = (new_f, state[0])

                elif state in open:
                    if new_f < fvalues[state]:
                        fathers[state] = (curr, action)
                        gvalues[state] = new_g
                        fvalues[state] = new_f
                        terminated[state] = ter

                        open[state] = (new_f, state[0])

                else:
                    if new_f < fvalues[state]:
                        fathers[state] = (curr, action)
                        gvalues[state] = new_g
                        fvalues[state] = new_f
                        terminated[state] = ter

                        open[state] = (new_f, state[0])
                        close.remove(state)

        return [], 0, 0


class AStarEpsilonAgent():
    def __init__(self) -> None:
        pass

    def h_msap(self, env: DragonBallEnv, s):
        def h_manheten(state1, state2):
            (x1, y1) = env.to_row_col(state1)
            (x2, y2) = env.to_row_col(state2)

            return np.abs(x1 - x2) + np.abs(y1 - y2)

        min = np.inf
        if (h_manheten(s, env.d1) < min and not s[1]):
            min = h_manheten(s, env.d1)

        if (h_manheten(s, env.d2) < min and not s[2]):
            min = h_manheten(s, env.d2)

        for state in range(env.nrow * env.ncol):
            if env.is_final_state((state, True, True)):
                if (h_manheten(s, (state, True, True)) < min):
                    min = h_manheten(s, (state, True, True))

        return min

    def f(self, env, state, g, w):
        return (1 - w) * g + w * self.h_msap(env, state)

    def next(self, env, open, epsilon, gvalues):
        focal = heapdict.heapdict()
        min_f = open.peekitem()[1][0]
        for state in open.keys():
            if open[state][0] <= min_f * (1.0 + epsilon):
                focal[state] = (gvalues[state], state[0])

        return focal.popitem()[0]

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        curr = env.get_initial_state()

        if env.is_final_state(curr):
            return [], 0, 0

        h_weight = 0.5
        open = heapdict.heapdict()
        close = []
        fathers = {curr: (None, -1)}
        gvalues = {curr: 0}
        terminated = {curr: False}

        expanded = 0

        fvalues = {}
        fvalues[curr] = self.f(env, curr, gvalues[curr], h_weight)
        open[curr] = (self.f(env, curr, gvalues[curr], h_weight), curr[0])

        while not len(open) == 0:
            curr = self.next(env, open, epsilon, gvalues)
            open.pop(curr)

            if env.is_final_state(curr):
                # Calculate track to the solution
                actions = []
                iterator = curr
                while fathers[iterator][1] != -1:
                    actions = [fathers[iterator][1]] + actions
                    iterator = fathers[iterator][0]

                return actions, gvalues[curr], expanded

            expanded += 1
            close.append(curr)

            if (terminated[curr]):
                continue

            for action, ((position, _, _), cost, ter) in env.succ(curr).items():
                state = (position, curr[1] or (position == env.d1[0]), curr[2] or (position == env.d2[0]))

                new_g = gvalues[curr] + cost
                new_f = self.f(env, state, new_g, h_weight)

                if (state not in close) and (state not in open.keys()):
                    fathers[state] = (curr, action)
                    gvalues[state] = new_g
                    fvalues[state] = new_f
                    terminated[state] = ter
                    open[state] = (fvalues[state], state[0])

                elif state in open.keys():
                    if new_f < fvalues[state]:
                        fathers[state] = (curr, action)
                        gvalues[state] = new_g
                        fvalues[state] = new_f
                        terminated[state] = ter

                        open[state] = (new_f, state[0])

                else:
                    if new_f < fvalues[state]:
                        fathers[state] = (curr, action)
                        gvalues[state] = new_g
                        fvalues[state] = new_f
                        terminated[state] = ter

                        open[state] = (fvalues[state], state[0])
                        close.remove(state)

        return [], 0, 0
