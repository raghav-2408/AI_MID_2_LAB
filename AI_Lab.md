# BFS Traversal
```python
def add_node(v):
    if v in graph:
        print(v, "already present in the graph")
    else:
        graph[v] = []

def add_edge(v1, v2):
    if v1 not in graph:
        print(v1, "is not present in the graph")
    elif v2 not in graph:
        print(v2, "is not present in the graph")
    else:
        graph[v1].append(v2)
        graph[v2].append(v1)

def bfs(node, visited, graph):
    visited.append(node)
    queue.append(node)
    while queue:
        m = queue.pop(0)
        print(m, end = " ")
        for i in graph[m]:
            if i not in visited:
                visited.append(i)
                queue.append(i)

queue = []
visited = []
graph = {}

add_node("5")
add_node("3")
add_node("7")
add_node("2")
add_node("4")
add_node("8")

add_edge("5", "3")
add_edge("5", "7")
add_edge("3", "2")
add_edge("3", "4")
add_edge("7", "8")
add_edge("4", "8")

# print(graph)
bfs("5", visited, graph)
```



# DFS Traversal

```python
def add_node(v):
    if v in graph:
        print(v, "not present in the graph")
    else:
        graph[v] = []

def add_edge(v1, v2):
    if v1 not in graph:
        print(v1, "not present in the graph")
    elif v2 not in graph:
        print(v2, "not present in the graph")
    else:
        graph[v1].append(v2)
        graph[v2].append(v1)

def dfs(node, visited, graph):
    if node not in graph:
        print(node, "not present in the graph")
    if node not in visited:
        print(node)
        visited.add(node)
        for i in graph[node]:
            dfs(i, visited, graph)

visited = set()
graph = {}

add_node("A")
add_node("B")
add_node("C")
add_node("D")
add_node("E")

add_edge("A", "B")
add_edge("A", "C")
add_edge("A", "D")
add_edge("B", "D")
add_edge("B", "E")
add_edge("C", "D")
add_edge("D", "E")

print(graph)
dfs("A", visited, graph)
```

# Graph Coloring 

```python
colors = ["red", "blue", "green", "yellow", "black"]
states = ["Andhra", "karnataka", "tamilnadu", "kerala"]
neighbors = {}
neighbors["Andhra"] = ["karnataka", "tamilnadu"]
neighbors["karnataka"] = ["Andhra", "tamilnadu", "kerala"]
neighbors["tamilnadu"] = ["Andhra", "kerala", "karnataka"]
neighbors["kerala"] = ["karnataka", "tamilnadu"]

col_of_s = {}

def promising(state, color):
    for neighbor in neighbors.get(state):
        temp = col_of_s.get(neighbor)
        if temp == color:
            return False
    return True

def get_color(state):
    for i in colors:
        if promising(state, i):
            return i
    return None

def main():
    for i in states:
        col_of_s [i] = get_color(i)
    print(col_of_s)

main()
```

# Travelling Sales Person

```python
from sys import maxsize
from itertools import permutations

v = 4

def tsp(graph, s):
    vertex = []
    for i in range(v):
        print(i)
        if i!=s:
            vertex.append(i)
    mn = maxsize
    next_p = permutations(vertex)

    for i in next_p:
        print(i)
        current_w = 0
        k = s
        for j in i:
            current_w += graph[k][j]
            k = j
        current_w += graph[k][s]
        mn = min(mn, current_w)
        print(mn)
    return mn

if __name__ == "__main__":
    graph = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
    s = 0
    print(tsp(graph, s))
```

# A* Search

```python
def astar(start, stop):
    openl = set(start)
    closel = set()
    g = {}
    parent = {}
    g[start] = 0
    parent[start] = start
    
    while len(openl) > 0:
        n = None
        for v in openl:
            if n==None or g[v] + h(v) < g[n] + h(n): # h -> heuristic
                n = v
        if n==stop and graph_nodes[n] == None:
            pass
        else:
            for m, weight in get_neigh(n):
                if m not in openl and m not in closel:
                    openl.add(m)
                    parent[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parent[m] = n
                        if m in closel:
                            closel.remove(m)
                            openl.add(m)
        if n==None:
            print("No path")
            return None
        if n==stop:
            path = []
            while parent[n] != n:
                path.append(n)
                n = parent[n]
                print(path)
            path.append(start)
            path.reverse()
            print(path)
            return path
        openl.remove(n)
        closel.add(n)
    print("No path ")
    return None
def get_neigh(v):
    if v in graph_nodes:
        return graph_nodes[v]
    else:   
        return None
def h(n):
    H_dist = {'A' : 11, 'B' : 6, 'C': 5, 'D': 7, 'E':3, 'F': 6, 'G':5, 'H':3, 'I':1, 'J':0}
    return H_dist[n]


graph_nodes = {'A': [('B', 6), ('F', 3)], 'B': [('A', 6), ('C', 3), ('D', 2)], 'C': [('B', 3), ('D', 1), ('E', 5)], 'D': [('B', 2), ('C', 1), ('E', 8)], 'E': [('C', 5), ('D', 8), ('I', 5), ('J', 5)], 'F': [('A', 3), ('G', 1), ('H', 7)], 'G': [('F', 1), ('I', 3)], 'H': [('F', 7), ('I', 2)], 'I': [('E', 5), ('G', 3), ('H', 2), ('J', 3)], 'J' : [('G', 5), ('H', 3)]}


astar('A', 'J')
```
# Missionaries and Cannibals
``` python
def is_valid(states):
    m1, c1, b, m2, c2 = states
    return 0<=m1<=3 and 0<=c1<=3 and 0<=m2<=3 and 0<=c2<=3 and (m1==0 or m1>=c1) and (m2==0 or m2>=c2)
# note  (m1==0 or m1>=c1) and (m2==0 or m2>=c2) -> dont add extra ()
def gen_next(states):
    m1, c1, b, m2, c2 = states
    nextt = []
    moves = [(1, 0),(2, 0), (0, 1), (0, 2), (1, 1)]
    for dm, dc in moves:
        if b==1:
            new = (m1-dm, c1-dc, 0, m2+dm, c2+dc)
        else:
            new = (m1+dm, c1+dc, 1, m2-dm, c2-dc)
        if is_valid(new):
            nextt.append(new)
    return nextt

def solve(start_state):
    queue = [(start_state, [start_state])]
    while queue:
        curr_stat, path = queue.pop(0) 
        if curr_stat == (0, 0, 0, 3, 3):
            return path
        for i in gen_next(curr_stat):
            if i not in path:
                queue.append((i, path + [i]))
    return None

start_state = (3, 3, 1, 0, 0)

solution = solve(start_state)

if solution:
    a = 0
    for state in solution:
        m1, c1, b, m2, c2 = state
        print(f"{m1} missionaries, {c1} cannibals {'boat on left' if b==1 else 'boat on right'} {m2} missionaries, {c2} cannibals")
        a += 1

    print("cost : ", a)
else:
    print("No path")    
```

## Mid - 2 Part

# Water Jug  

```python
from collections import defaultdict

jug1 = int(input())
jug2 = int(input())
target = int(input())
visited = defaultdict(lambda: False)

def wjp(amt1, amt2):
    if (amt1 == target and amt2 == 0) or (amt1 == 0 and amt2 == target):
        print(amt1, amt2)
        return True
    if visited[(amt1, amt2)] == False:
        print(amt1, amt2)
        visited[(amt1, amt2)] = True
        return (
            wjp(amt1, 0) or
            wjp(0, amt2) or
            wjp(jug1, amt2) or
            wjp(amt1, jug2) or
            wjp(amt1 + min(amt2, (jug1 - amt1)), amt2 - min(amt2, (jug1 - amt1))) or
            wjp(amt1 - min(amt1, (jug2 - amt2)), amt2 + min(amt1, (jug2 - amt2)))
        )
    else:
        return False
wjp(0, 0)
```

```python
Output :

0 0
4 0
4 3
0 3
3 0
3 3
4 2
0 2
```

# Hangman Game

`Note : Also create a hangman.py file in the same directory`

```python
import random
from hangman import li
a = ["apple", "mango", "grapes"]
word = random.choice(a)
temp = []
lives = 6
for i in range(len(word)):
    temp.append("_")
print(temp)
game = False
while not game:
    guess = input("Guess :")
    for i in range(len(word)):
        if guess == word[i]:
            temp[i] = guess
    print(temp)
    if guess not in word:
        lives-=1
        if lives==0:
            print("You lose")
            game = True
    if '_' not in temp:
        print("You win")
        game = True
    print(li[lives])
``` 

# Tic Tac Toe

```python
import os
import time

board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
player = 1

####Win Flags####
Win = 1
Draw = -1
Running = 0
Stop = 1
##############
Game = Running
Mark = 'X'

#This Function Draws Game Board
def DrawBoard():
    print(" %c | %c | %c " % (board[1], board[2], board[3]))
    print("___|___|___")
    print(" %c | %c | %c " % (board[4], board[5], board[6]))
    print("___|___|___")
    print(" %c | %c | %c " % (board[7], board[8], board[9]))
    print("  |   |  ")

#This function Checks position is empty or not
def CheckPosition(x):
    if board[x] == ' ':
        return True
    else:
        return False

#This Function Checks player has won or not
def CheckWin():
    global Game
    #Horizontal winning condition
    if board[1] == board[2] and board[2] == board[3] and board[1] != ' ':
        Game = Win
    elif board[4] == board[5] and board[5] == board[6] and board[4] != ' ':
        Game = Win
    elif board[7] == board[8] and board[8] == board[9] and board[7] != ' ':
        Game = Win
    #Vertical Winning Condition
    elif board[1] == board[4] and board[4] == board[7] and board[1] != ' ':
        Game = Win
    elif board[2] == board[5] and board[5] == board[8] and board[2] != ' ':
        Game = Win
    elif board[3] == board[6] and board[6] == board[9] and board[3] != ' ':
        Game = Win
    #Diagonal Winning Condition
    elif board[1] == board[5] and board[5] == board[9] and board[5] != ' ':
        Game = Win
    elif board[3] == board[5] and board[5] == board[7] and board[5] != ' ':
        Game = Win
    #match Tie or Draw Condition
    elif board[1] != ' ' and board[2] != ' ' and board[3] != ' ' and 
            board[4] != ' ' and board[5] != ' ' and board[6] != ' ' and 
            board[7] != ' ' and board[8] != ' ' and board[9] != ' ':
        Game = Draw
    else:
        Game = Running

print("Tic-Tac-Toe Game")
print("Player 1 [X] --- Player 2 [O]\n")
print()
print()
print("Please Wait...")
time.sleep(3)
while Game == Running:
    os.system('cls')
    DrawBoard()
    if player % 2 != 0:
        print("Player 1's chance")
        Mark = 'X'
    else:
        print("Player 2's chance")
        Mark = 'O'
    choice = int(input("Enter the position between [1-9] where you want to mark: "))
    if CheckPosition(choice):
        board[choice] = Mark
        player += 1
        CheckWin()
    os.system('cls')
    DrawBoard()
    if Game == Draw:
        print("Game Draw")
    elif Game == Win:
        player -= 1
        if player % 2 != 0:
            print("Player 1 Won")
        else:
            print("Player 2 Won")

```


# 8 Queens

```python
N = int(input())
board = [[0]*N for _ in range(N)]

def attack(i, j):
    # row, columns
    for k in range(N):
        if board[i][k] == 1 or board[k][j] == 1:
            return True
    # diagonals
    for k in range(N):
        for l in range(N):
            if k+l == i+j or k-l == i-j:
                if board[k][l] == 1:
                    return True
    return False

def nQueens(n):
    if n==0:
        return True
    for i in range(N):
        for j in range(N):
            if not(attack(i, j)) and board[i][j]!=1:
                board[i][j] = 1
                if nQueens(n-1):
                    return True
                board[i][j] = 0
    return False

nQueens(N)
for i in board:
    print(i)
```
# Bayesian Network

`add modules manually`
```python
from pomegranate import bayesian_network, state, ConditionalProbabilityTable, DiscreteDistribution

guest = DiscreteDistribution({'A' : 1/3, 'B' : 1/3, 'C':1/3})
prize = DiscreteDistribution({'A' : 1/3, 'B' : 1/3, 'C':1/3})
monty = ConditionalProbabilityTable(

    [['A', 'A', 'A', 0.5],
    
    # < ---------- >
    # add .............
    # < ---------- >

    ['C', 'C', 'C', 0.0]], [guest, prize]
)

s1 = state(guest, name = "guest")
s2 = state(prize, name = "prize")
s3 = state(monty, name = "monty")

model = bayesian_network("Monty Hall Problem")
print(model)

model.add_states(s1, s2, s3)
model.add_edges(s1, s3)
model.bake()

print(model.probability([['', '', ''], ['', '', ''], ['', '', ''], ['', '', '']]))
print(model.predict([['', 'None', ''], ['', '', 'None'], ['None', '', ''], ['', 'None', '']]))

```
# Hidden Markov Model

```python
import itertools 
import pandas as pd

states = ['sleeping', 'eating', 'pooping']

hidden_states = ['healthy', 'sick']
pi = [0.5, 0.5]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
 
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]                 
a_df.loc[hidden_states[1]] = [0.4, 0.6]
 
print(a_df)
 
observable_states = states
 
b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]
 
print(b_df)
 
def HMM(obsq,b_df,a_df,pi,states,hidden_states):
        hidst=list(itertools.combinations_with_replacement(hidden_states,len(obsq)))
        sum=0
        for k in hidst:
                prod=1
                for j in range(len(k)):
                    for i in obsq:
                        c=0
                        if c==0:
                            prod*=b_df[i][k[j]]*pi[hidden_states.index(k[j])]
                            c=1
                        else:
                            prod*=a_df[k[j]][k[j-1]]*b_df[i][k[j]]
                sum+=prod
                c=0
        return sum
 
def vertibi(obsq,b_df,a_df,pi,states,hidden_states):
        sum=0
        hidst=list(itertools.combinations_with_replacement(hidden_states,len(obsq)))
        for k in hidst:
                sum1=0
                prod=1
                for j in range(len(k)):
                    for i in obsq:
                        c=0
                        if c==0:
                            prod*=b_df[i][k[j]]*pi[hidden_states.index(k[j])]
                            c=1
                        else:
                            prod*=a_df[k[j]][k[j-1]]*b_df[i][k[j]]
                c=0
                sum1+=prod
                if(sum1>sum):
                    sum=sum1
                    hs=k
        return sum,hs
 
obsq=['pooping','pooping','pooping']
print(HMM(obsq,b_df,a_df,pi,states,hidden_states))
print(vertibi(obsq,b_df,a_df,pi,states,hidden_states))
```

