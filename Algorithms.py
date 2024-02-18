import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


# class node():
#     def __init__(self, state, father=None):
#         self.state = state
#         self.father = father
#         self.children = []
#
#     def expand(self, env):
#         print(env.succ(self.state).values())
#         for (state, cost, terminated) in env.succ(self.state).values():
#             if (cost != np.inf):
#                 self.children.append((state, cost, terminated))


class BFSAgent():

    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        curr = env.get_state()

        if env.is_final_state(curr):
            return [], 0, 0

        open = [curr]
        close = []
        fathers = {curr: (None, -1)}
        costs = {curr: 0}

        while not len(open) == 0:
            curr = open[0]
            open = open[1:]
            close.append(curr)

            for action, succ in env.succ(curr).items():
                if succ[0] is not None:
                    ((position, _, _), cost, _) = succ

                    state = (position, curr[1] or (position == env.d1[0]), curr[2] or (position == env.d2[0]))

                    if (state not in close) and (state not in open):
                        if env.is_final_state(state):
                            #Calculate track to the solution
                            actions = [action]
                            iterator = curr
                            while fathers[iterator][1] != -1:
                                actions = [fathers[iterator][1]] + actions
                                iterator = fathers[iterator][0]

                            return actions, costs[curr] + cost, len(close)

                        open.append(state)
                        fathers[state] = (curr, action)
                        costs[state] = costs[curr] + cost

        return [], -1, -1


class WeightedAStarAgent():

    def h_msap(self, env: DragonBallEnv, s):
        def h_manheten(state1, state2):
            (x1, y1) = env.to_row_col(state1)
            (x2, y2) = env.to_row_col(state2)

            return np.abs(x1 - x2) + np.abs(y1 - y2)

        min = h_manheten(s, env.d1)

        if(h_manheten(s, env.d2) < min):
            min = h_manheten(s, env.d2)

        for state in range (env.nrow * env.ncol):
            if env.is_final_state((state, True, True)):
                if (h_manheten(s, (state, True, True)) < min):
                    min = h_manheten(s, (state, True, True))

        return min

    def f(self, env, state, g, w):
        return (1 - w) * g + w * self.h_msap(env, state)

    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        curr = env.get_state()

        if env.is_final_state(curr):
            return [], 0, 0

        open = heapdict.heapdict()
        close = []
        fathers = {curr: (None, -1)}
        gvalues = {curr: 0}

        fvalues = {}
        fvalues[curr] = self.f(env, curr, gvalues[curr], h_weight)
        open[curr] = self.f(env, curr, gvalues[curr], h_weight)

        while not len(open) == 0:
            curr = open.popitem()[0]
            close.append(curr)

            if env.is_final_state(curr):
                # Calculate track to the solution
                actions = []
                iterator = curr
                while fathers[iterator][1] != -1:
                    actions = [fathers[iterator][1]] + actions
                    iterator = fathers[iterator][0]

                return actions, gvalues[curr], len(close)

            for action, succ in env.succ(curr).items():
                if succ[0] is not None:
                    ((position, _, _), cost, _) = succ
                    state = (position, curr[1] or (position == env.d1[0]), curr[2] or (position == env.d2[0]))

                    new_g = gvalues[curr] + cost
                    new_f = self.f(env, state, new_g, h_weight)

                    if (state not in close) and (state not in open.keys()):
                        fathers[state] = (curr, action)
                        gvalues[state] = new_g
                        fvalues[state] = new_f
                        open[state] = fvalues[state]

                    elif state in open.keys():
                        if new_f < fvalues[state]:
                            fathers[state] = (curr, action)
                            gvalues[state] = new_g
                            fvalues[state] = new_f

                    else:
                        if new_f < fvalues[state]:
                            fathers[state] = (curr, action)
                            gvalues[state] = new_g
                            fvalues[state] = new_f

                            open[state] = fvalues[state]
                            close.remove(state)

        return [], -1, -1

class AStarEpsilonAgent():
    def __init__(self) -> None:
        pass

    def h_msap(self, env: DragonBallEnv, s):
        def h_manheten(state1, state2):
            (x1, y1) = env.to_row_col(state1)
            (x2, y2) = env.to_row_col(state2)

            return np.abs(x1 - x2) + np.abs(y1 - y2)

        min = h_manheten(s, env.d1)

        if(h_manheten(s, env.d2) < min):
            min = h_manheten(s, env.d2)

        for state in range (env.nrow * env.ncol):
            if env.is_final_state((state, True, True)):
                if (h_manheten(s, (state, True, True)) < min):
                    min = h_manheten(s, (state, True, True))

        return min

    def f(self, env, state, g, w):
        return (1 - w) * g + w * self.h_msap(env, state)


    def next(self, env, open, epsilon, gvalues):
        min_f = open.peekitem()[1]

        focal = [state for state in open.keys() if open[state] <= min_f * (1 + epsilon)]

        """
        min_h = np.inf
        next_state = None
        for state in focal:
            if self.h_msap(env, state) < min_h:
                min_h = self.h_msap(env, state)
                next_state = state

        """
        min_g = np.inf
        next_state = None
        for state in focal:
            if gvalues[state] < min_g:
                min_g = gvalues[state]
                next_state = state

        return next_state

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        curr = env.get_state()

        if env.is_final_state(curr):
            return [], 0, 0

        h_weight = 0.5
        open = heapdict.heapdict()
        close = []
        fathers = {curr: (None, -1)}
        gvalues = {curr: 0}

        fvalues = {}
        fvalues[curr] = self.f(env, curr, gvalues[curr], h_weight)
        open[curr] = self.f(env, curr, gvalues[curr], h_weight)

        while not len(open) == 0:
            curr = self.next(env, open, epsilon, gvalues)
            open.pop(curr)
            close.append(curr)

            if env.is_final_state(curr):
                # Calculate track to the solution
                actions = []
                iterator = curr
                while fathers[iterator][1] != -1:
                    actions = [fathers[iterator][1]] + actions
                    iterator = fathers[iterator][0]

                return actions, gvalues[curr], len(close)

            for action, succ in env.succ(curr).items():
                if succ[0] is not None:
                    ((position, _, _), cost, _) = succ
                    state = (position, curr[1] or (position == env.d1[0]), curr[2] or (position == env.d2[0]))

                    new_g = gvalues[curr] + cost
                    new_f = self.f(env, state, new_g, h_weight)

                    if (state not in close) and (state not in open.keys()):
                        fathers[state] = (curr, action)
                        gvalues[state] = new_g
                        fvalues[state] = new_f
                        open[state] = fvalues[state]

                    elif state in open.keys():
                        if new_f < fvalues[state]:
                            fathers[state] = (curr, action)
                            gvalues[state] = new_g
                            fvalues[state] = new_f

                    else:
                        if new_f < fvalues[state]:
                            fathers[state] = (curr, action)
                            gvalues[state] = new_g
                            fvalues[state] = new_f

                            open[state] = fvalues[state]
                            close.remove(state)

        return [], -1, -1
