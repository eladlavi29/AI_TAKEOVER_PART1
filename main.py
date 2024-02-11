import time
from IPython.display import clear_output



from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from Algorithms import *
BFS_agent = BFSAgent()
actions, total_cost, expanded = BFS_agent.search(env)
print(f"Total_cost: {total_cost}")
print(f"Expanded: {expanded}")
print(f"Actions: {actions}")

assert total_cost == 119.0, "Error in total cost returned"