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
    "8x8": ['SAHTFHAHTHAFLTTHHHLF', 'HLAHFHALAHHHFLLGTATL', 'FATFGFDLFTAFHAFLTAAT', 'DFHTTTFHTFFHHFLFFAAH', 'LTTFLHLTTTHHHTTHFALF', 'TLLTTLTHHAAFALLFATAH', 'FLHLLHAAHLAATLAAALAH', 'FTLHLHLATLALFFHFALHT', 'ALHAALALHALHAFTTGLAT', 'HTHTTFAATLLATLFHHLTA'],
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
sold = [(0, False, False), (1, False, False), (21, False, False), (41, False, False), (40, False, False), (61, False, False), (42, False, False), (43, False, False), (44, False, False), (22, False, False), (81, False, False), (80, False, False), (63, False, False), (64, False, False), (100, False, False), (84, False, False), (65, False, False), (45, False, False), (66, False, False), (101, False, False), (82, False, False), (104, False, False), (46, True, False), (86, False, False), (102, False, False), (121, False, False), (45, True, False), (44, True, False), (120, False, False), (83, False, False), (103, False, False), (104, False, False), (60, False, True), (40, False, True), (41, False, True), (42, False, True), (43, False, True), (44, False, True), (22, False, True), (63, False, True), (64, False, True), (61, False, True), (105, False, False), (106, False, False), (86, False, False), (87, False, False), (124, False, False), (21, False, True), (22, False, True), (84, False, True), (47, True, False), (26, True, False), (65, False, True), (126, False, False), (141, False, False), (45, False, True), (65, True, False), (64, True, False), (66, False, True), (123, False, False), (124, False, False), (140, False, False), (80, False, True), (81, False, True), (82, False, True), (83, False, True), (84, False, True), (64, False, True), (44, False, True), (63, False, True), (65, False, True), (45, False, True), (66, False, True), (104, False, True), (46, True, True), (86, False, True), (102, False, True), (103, False, True), (104, False, True), (84, False, True), (64, False, True), (44, False, True), (65, False, True), (45, False, True), (66, False, True), (46, True, True), (86, False, True), (45, True, True)]
elad = [(0, False, False), (1, False, False), (21, False, False), (41, False, False), (40, False, False), (61, False, False), (42, False, False), (43, False, False), (44, False, False), (22, False, False), (81, False, False), (80, False, False), (63, False, False), (64, False, False), (100, False, False), (84, False, False), (65, False, False), (45, False, False), (66, False, False), (101, False, False), (82, False, False), (46, True, False), (102, False, False), (121, False, False), (45, True, False), (44, True, False), (120, False, False), (83, False, False), (103, False, False), (104, False, False), (60, False, True), (40, False, True), (41, False, True), (42, False, True), (43, False, True), (61, False, True), (105, False, False), (106, False, False), (86, False, False), (87, False, False), (21, False, True), (22, False, True), (47, True, False), (26, True, False), (126, False, False), (141, False, False), (65, True, False), (64, True, False), (123, False, False), (124, False, False), (140, False, False), (80, False, True), (81, False, True), (82, False, True), (83, False, True), (63, False, True), (102, False, True), (103, False, True), (104, False, True), (84, False, True), (64, False, True), (44, False, True), (65, False, True), (45, False, True), (66, False, True), (46, True, True), (86, False, True), (45, True, True)]

temp = []
for i in elad:
    if i not in sold:
        temp.append(i)

print(temp)

env = DragonBallEnv(MAPS["8x8"])

env.reset()

BFS_agent = BFSAgent()
actions, total_cost, expanded = BFS_agent.search(env)
print(f"BFS-G:")
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")
print("\n");

#print_solution(actions, env)

WA_agent = WeightedAStarAgent()

actions, total_cost, expanded = WA_agent.search(env, h_weight=0.9)
print(f"wA*:")
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")
print("\n");

AStarEpsilon_agent = AStarEpsilonAgent()
actions, total_cost, expanded = AStarEpsilon_agent.search(env, epsilon=1)
print(f"epsilon A*:")
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")
print("\n");

import csv
from rosman_nimrod_maps import small, medium, large, segel

final = {}
final.update(small)
final.update(medium)
final.update(large)
final.update(segel)
test_envs = {}
for board_name, board in final.items():
    test_envs[board_name] = DragonBallEnv(board)

test_expanded = True
result = "results.csv" if test_expanded else "results_without_expanded.csv"
user_result = "my_results.csv"
BFS_agent = BFSAgent()
WAStar_agent = WeightedAStarAgent()
AStar_epsilon_agent = AStarEpsilonAgent()
weights = [0.1, 0.3, 0.5, 0.7, 0.9]
epsilons = weights.copy()
agents_search_function = [
    BFS_agent.search,
]

with open(user_result, 'w') as f:
    writer = csv.writer(f)
    for env_name, env in test_envs.items():
        data = [env_name]
        for agent in agents_search_function:
            actions, total_cost, expanded = agent(env)
            data += [total_cost, expanded, actions]
        for w in weights:
            actions, total_cost, expanded = WAStar_agent.search(env, w)
            if test_expanded:
                data += [total_cost, expanded, actions]
            else:
                data += [total_cost, actions]
        for espilon in epsilons:
            actions, total_cost, expanded = AStar_epsilon_agent.search(env, espilon)
            if test_expanded:
                data += [total_cost, expanded, actions]
            else:
                data += [total_cost, actions]

        writer.writerow(data)


def compare_csv(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        for row1, row2 in zip(reader1, reader2):
            if row1 != row2:
                return False
        return True


if compare_csv(result, user_result):
    print(f'Congrats, You have passed!')
else:
    print(f'Give it another try :)')
