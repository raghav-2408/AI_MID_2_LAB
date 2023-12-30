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
        Ga
        me = Win
    elif board[3] == board[5] and board[5] == board[7] and board[5] != ' ':
        Game = Win
    #match Tie or Draw Condition
    elif board[1] != ' ' and board[2] != ' ' and board[3] != ' ' and \
            board[4] != ' ' and board[5] != ' ' and board[6] != ' ' and \
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

