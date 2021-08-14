import numpy as np
from queue import Queue
from queue import LifoQueue
import time
import heapq


def find_zero(puzzel):
    zero_index = np.where(puzzel == 0)
    return int(str(zero_index[0])[1])


def change_position(node, explored, frontier):
    zero_index = find_zero(node)
    temp_puzzle = node.copy()
    parent_depth = explored[str(temp_puzzle)]
    if zero_index - 3 >= 0:
        swap(temp_puzzle, zero_index, zero_index - 3)
        if str(temp_puzzle) not in explored.keys():
            frontier.put(temp_puzzle)
            explored[str(temp_puzzle)] = parent_depth + 1
        temp_puzzle = node.copy()

    if zero_index + 3 <= 8:
        swap(temp_puzzle, zero_index, zero_index + 3)
        if str(temp_puzzle) not in explored.keys():
            frontier.put(temp_puzzle)
            explored[str(temp_puzzle)] = parent_depth + 1
        temp_puzzle = node.copy()
    if zero_index - 1 >= 0 and zero_index != 3 and zero_index != 6:
        swap(temp_puzzle, zero_index, zero_index - 1)
        if str(temp_puzzle) not in explored.keys():
            frontier.put(temp_puzzle)
            explored[str(temp_puzzle)] = parent_depth + 1
        temp_puzzle = node.copy()
    if zero_index + 1 <= 8 and zero_index != 2 and zero_index != 5:
        swap(temp_puzzle, zero_index, zero_index + 1)
        if str(temp_puzzle) not in explored.keys():
            frontier.put(temp_puzzle)
            explored[str(temp_puzzle)] = parent_depth + 1


def swap(temp_puzzle, zero_index, number_index):
    temp_puzzle[zero_index] = temp_puzzle[number_index]
    temp_puzzle[number_index] = 0


def bfs(puzzel):
    nodes_expanded = 0
    frontier = Queue()
    frontier.put(puzzel)
    explored = {}
    explored[str(puzzel)] = 0
    while not frontier.empty():
        node = frontier.get()
        display(node)
        if finish(node):
            break
        nodes_expanded += 1
        change_position(node, explored, frontier)
    print('Output')
    display(node)
    print(f'Expanded nodes : {nodes_expanded}')
    print(f'Depth: {explored[str(node)]}')


def dfs(puzzel):
    nodes_expanded = 0
    frontier = LifoQueue()
    frontier.put(puzzel)
    explored = {}
    explored[str(puzzel)] = 0
    while not frontier.empty():
        node = frontier.get()
        display(node)
        if finish(node):
            break
        nodes_expanded += 1
        change_position(node, explored, frontier)
    print('Output')
    display(node)
    print(f'Expanded nodes : {nodes_expanded}')
    print(f'Depth: {explored[str(node)]}')


def finish(puzzel):
    if str(puzzel) == str(np.array(range(9))):
        return True
    else:
        return False


def display(node):
    print(f'| {node[0]} | {node[1]} | {node[2]} |')
    print(f'| {node[3]} | {node[4]} | {node[5]} |')
    print(f'| {node[6]} | {node[7]} | {node[8]} |')
    print()


def swapAstar(state, i, j):
    new_state = np.array(state.value)
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state


def manhattan(state):
    state = getindex(state)
    goal = np.array([0,1,2,3,4,5,6,7,8])
    s = sum((abs(state // 3 - goal // 3) + abs(state % 3 - goal % 3))[1:])  # getting the sum of all x's and y's for all tiles ignore zero tile
    return s


def euclidean(state):
    state = getindex(state)
    goal = np.array([0,1,2,3,4,5,6,7,8])
    s = sum((((state // 3 - goal // 3)**2 + (state % 3 - goal % 3)**2)**0.5)[1:]) # get the straight line distance between the tile's position in any state and
                                                                                  # its position in goal state
    return s


def getindex(state):
    index = np.array([0,1,2,3,4,5,6,7,8])
    for x, y in enumerate(state):
        index[y] = x
    return index


def get_parents(solution):
    current = solution
    chain = [current]
    while current.parent is not None:
        chain.append(current.parent)
        current = current.parent
    return chain

def path(solution):
    path = [node.movement for node in get_parents(solution)[-2::-1]]   # [-2::-1] ignore the initial state (2nd level -> goal)
    return path, [node for node in get_parents(solution)[-1::-1]]   # all states (initial -> goal)


class State:   # this class for collecting all attributes of state in game
    heuristic_cost = None
    heuristic_type = None
    parent = None
    value = None
    movement = None
    depth = int()
    zero = None
    cost = int()

    def __init__(self, value, parent=None, movement=None, zero = None, depth=0, heuristic_cost=None, heuristic_type=None):
        self.parent = parent
        self.value = np.array(value)
        self.movement = movement
        self.depth = depth
        self.zero = zero
        self.heuristic_type = heuristic_type
        self.heuristic_cost = heuristic_cost
        if (self.heuristic_type == "manhattan" or self.heuristic_type == "euclidean") and self.heuristic_cost != None:
            self.cost = self.depth + self.heuristic_cost

    def __lt__(self, other):   # overriding this method for using it for comparison in heap
        return self.cost < other.cost


def get_neighbors(state):
    neighbors = []

    if state.zero > 2:  # check for up child
        new_state = swapAstar(state, state.zero, state.zero - 3)

        if state.heuristic_type == "manhattan":
            Up = State(new_state, state, 'Up', find_zero(new_state), state.depth + 1, manhattan(new_state), "manhattan")
            neighbors.append(Up)
        elif state.heuristic_type == "euclidean":
            Up = State(new_state, state, 'Up', find_zero(new_state), state.depth + 1, euclidean(new_state), "euclidean")
            neighbors.append(Up)
        else:
            Up = State(new_state, state, 'Up', find_zero(new_state), state.depth + 1, None, None)
            neighbors.append(Up)

    if state.zero < 6: # check for down child
        new_state = swapAstar(state, state.zero, state.zero + 3)
        if state.heuristic_type == "manhattan":
            Down = State(new_state, state, 'Down', find_zero(new_state), state.depth + 1, manhattan(new_state), "manhattan")
            neighbors.append(Down)
        elif state.heuristic_type == "euclidean":
            Down = State(new_state, state, 'Down', find_zero(new_state), state.depth + 1, euclidean(new_state),
                         "euclidean")
            neighbors.append(Down)
        else:
            Down = State(new_state, state, 'Down', find_zero(new_state), state.depth + 1, None, None)
            neighbors.append(Down)

    if state.zero % 3 != 0:  # not in the first colomn and check for left child

        new_state = swapAstar(state, state.zero, state.zero - 1)
        if state.heuristic_type == "manhattan":
            Left = State(new_state, state, 'Left', find_zero(new_state), state.depth + 1, manhattan(new_state), "manhattan")
            neighbors.append(Left)
        elif state.heuristic_type == "euclidean":
            Left = State(new_state, state, 'Left', find_zero(new_state), state.depth + 1, euclidean(new_state), "euclidean")
            neighbors.append(Left)
        else:
            Left = State(new_state, state, 'Left', find_zero(new_state), state.depth + 1, None, None)
            neighbors.append(Left)

    if (state.zero + 1) % 3 != 0:  # not in the last colomn and check for right child

        new_state = swapAstar(state, state.zero, state.zero + 1)
        if state.heuristic_type == "manhattan":
            Right = State(new_state, state, 'Right', find_zero(new_state), state.depth + 1, manhattan(new_state), "manhattan")
            neighbors.append(Right)
        elif state.heuristic_type == "euclidean":
            Right = State(new_state, state, 'Right', find_zero(new_state), state.depth + 1, euclidean(new_state), "euclidean")
            neighbors.append(Right)
        else:
            Right = State(new_state, state, 'Right', find_zero(new_state), state.depth + 1, None, None)
            neighbors.append(Right)

    return neighbors


def show_state(state):
    if state.movement == 'Up': # check the movement of state's zero for printing in traceable format
        zero = find_zero(state.value)
        print('After moving ' + str(state.value[zero + 3]) + ' into empty space')
        display(state.value)

    elif state.movement == 'Down':
        zero = find_zero(state.value)
        print('After moving ' + str(state.value[zero - 3]) + ' into empty space')
        display(state.value)

    elif state.movement == 'Left':
        zero = find_zero(state.value)
        print('After moving ' + str(state.value[zero + 1]) + ' into empty space')
        display(state.value)

    elif state.movement == 'Right':
        zero = find_zero(state.value)
        print('After moving ' + str(state.value[zero - 1]) + ' into empty space')
        display(state.value)

    else:
        print('Initial State')
        display(state.value)


def A_star(state):
    nodes_expanded = 0
    max_depth = 0
    no = 0

    frontier = []
    explored = set()
    heapq.heappush(frontier, state)
    while frontier:
        state = heapq.heappop(frontier)
        explored.add(tuple(state.value))

        if finish(state.value):
            return state, max_depth, nodes_expanded

        for neighbor in get_neighbors(state):
            max_depth = max(max_depth, neighbor.depth)

            if tuple(neighbor.value) not in explored and list(neighbor.value) not in [list(node.value) for node in frontier]:
                heapq.heappush(frontier, neighbor)
                no +=1

            elif list(neighbor.value) in [list(node.value) for node in frontier]:
                index = [x for x, node in enumerate(frontier) if list(node.value) == list(neighbor.value)]

                if frontier[index[0]].cost < neighbor.cost:
                    frontier.pop(index[0])
                    heapq.heappush(frontier, neighbor)

        nodes_expanded = nodes_expanded + no
        no = 0

    return None


puzzel = np.array([1,4,2,6,5,8,7,3,0])
x = input('1) BFS\n2) DFS\n3) A*\n')

if int(x) == 1:
    start = time.time()
    bfs(puzzel)
    end = time.time()
elif int(x) == 2:
    start = time.time()
    dfs(puzzel)
    end = time.time()
elif int(x) == 3:
    s = None
    heuristic = input('select the type of heuristic:\n1) manhattan\n2) euclidean\n')
    print('\n')

    if int(heuristic) == 1:
        s = State(puzzel, None, None, find_zero(puzzel), 0, manhattan(puzzel), "manhattan")
    elif int(heuristic) == 2:
        s = State(puzzel, None, None, find_zero(puzzel), 0, euclidean(puzzel), "euclidean")

    start = time.time()
    sol = A_star(s)
    end = time.time()

    if sol is not None:
        move, nodes_path = path(sol[0])
        for node in nodes_path:
            show_state(node)

        print('The Path to Target: ' + " -> ".join(move))
        print("nodes_expanded: " + str(sol[2]))
        print('cost_of_path: ' + str(len(move)))
        print('search_depth: ' + str(sol[0].depth))
        print('max_search_depth: ' + str(sol[1]))
        print(f'Running time: {end - start}')


