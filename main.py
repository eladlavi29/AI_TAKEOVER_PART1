import time
from IPython.display import clear_output
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from Algorithms import *

MAPS = {
    "4x4": ["SFFF",
            "FDFF",
            "FFFD",
            "FFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFTAL",
        "TFFHFFTF",
        "FFFFFHTF",
        "FAFHFFFF",
        "FHHFFFHF",
        "DFTFHDTL",
        "FLFHFFFG",
    ],
}


def print_solution(actions, env: DragonBallEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
        state, cost, terminated = env.step(action)
        total_cost += cost
        clear_output(wait=True)

        print(env.render())
        print(f"Timestep: {i + 2}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Cost: {cost}")
        print(f"Total cost: {total_cost}")

        time.sleep(1)

        if terminated is True:
            break

#Main Test
env = DragonBallEnv(MAPS["8x8"])

env.reset()

# BFS_agent = BFSAgent()
# actions, total_cost, expanded = BFS_agent.search(env)
# print(f"Total_cost: {total_cost}")
# print(f"Expanded: {expanded}")
# print(f"Actions: {actions}")
#
# assert total_cost == 119.0, "Error in total cost returned"

#print_solution(actions, env)

#WA_agent = WeightedAStarAgent()

#actions, total_cost, expanded = WA_agent.search(env, h_weight=0.5)
#print(f"Total_cost: {total_cost}")
#print(f"Expanded: {expanded}")
#print(f"Actions: {actions}")

AStarEpsilon_agent = AStarEpsilonAgent()
actions, total_cost, expanded = AStarEpsilon_agent.search(env, epsilon=100)
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")

#print_solution(actions, env)


