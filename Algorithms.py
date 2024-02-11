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
        close = []
        curr = env.get_state()

        open = [curr]
        fathers = {curr: (None, -1)}
        costs = {curr: 0}

        while not len(open) == 0:
            curr = open[0]
            open = open[1:]
            close.append(curr)

            for action, ((position, _, _), cost, _) in env.succ(curr).items():
                if cost != np.inf:
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
                        print(state)

        return [], -1, -1


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
