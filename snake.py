from math import log2
from queue import Queue
import random


# 0 Empty
# 1 Snake
# 2 Food


def GetDirection(x1, y1, x2, y2):
    if x1 == x2:
        if y1 > y2:
            return 3
        else:
            return 4
    else:
        if x1 > x2:
            return 1
        else:
            return 2

def MakeBoard(size): 
    return [[0 for i in range(size)] for j in range(size)]

class Game:
    def __init__(self, board_size, snakeSize):
        self.board = MakeBoard(board_size)
        self.snake = Snake(snakeSize)
        self.size = len(self.board)
        self.fruit = None
        self.moveCount = 0
        self.score = 3
        self.prevMoves = []
        self.UpdateBoard()
        self.PlaceFood()
        
    def UpdateBoard(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.fruit:
                    self.board[i][j] = "🍎"
                elif (i, j) in self.snake.body:
                    self.board[i][j] = "🐍"
                else:
                    self.board[i][j] = "⬛"

    def ShowBoard(self):
        for row in self.board:
            print(row)
        print()


    def PlaceFood(self):
        available = []
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) not in self.snake.body:
                    available.append((i,j))
        if len(available) == 0:
            raise Exception("No available space")
        x, y = random.choice(available)
        self.fruit = (x, y)
        self.UpdateBoard()
        return x, y

    def isLegalMove(self, x, y):
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False
        if (x, y) in self.snake.body:
            return False
        return True

    def GetMoves(self):
        x, y = self.snake.body[0]
        moves = []
        possible_moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        for move in possible_moves:
            if self.isLegalMove(*move):
                moves.append(move)
        return moves

# Lets check if a sequence of moves equal to snake length is in the buffer
# We only need to worry about above 4 because we punish the snake severely if it does not eat the fruit
    def CheckRepeatMove(self):
        if len(self.prevMoves) < self.size ** 2:
            return False
        prevChunks = [self.prevMoves[i:i + 4] for i in range(0, len(self.prevMoves), 4)]
        seen_patterns = set()
        for chunk in prevChunks:
            if len(chunk) < 4:
                break
            # Create a unique string representation for each chunk
            chunk_pattern = '-'.join(f"{x},{y}" for x, y in chunk)
            if chunk_pattern in seen_patterns:
                print("Found a loop")
                print(seen_patterns, chunk_pattern)
                return True
            seen_patterns.add(chunk_pattern)
        return False

    

                

    


    def isGameOver(self):
        if(len(self.GetMoves()) == 0):
            if len(self.snake.body) <= 4:
                return True
            return True
        return False
    
    def WhereFruit(self):
        return self.fruit
    
    def Play(self, x, y, defaultSnakeSize=3, silent=True):
        
        if (x,y) not in self.GetMoves():
            raise Exception("Illegal move")
        if not self.isGameOver():
            if self.snake.Move(x, y, self.board) == "Fruit":
                self.PlaceFood()
                if not silent:
                    self.ShowBoard()
            self.UpdateBoard()
            self.prevMoves.append(self.snake.body[0])
            self.moveCount += 1
            # self.ShowBoard()

            
        return self.GetState(), self.GetScore(defaultSnakeSize), self.isGameOver()
    
    def GetScore(self, defaultSnakeSize):
        # We don't want to store any more moves than necessary to check for loops
        if len(self.prevMoves) > self.size ** 2:
            # Clear array because we don't need to check for very complicated loops just simple ones
            self.prevMoves = []

        f1 = pow(len(self.snake.body), 3)

        if self.CheckRepeatMove() or self.moveCount >= f1:
            print("Move Count Exceeded, Or Loop Detected")
            return -100000

        if self.isGameOver():
            if(len(self.snake.body) <= defaultSnakeSize):
                return -100000
            if(self.GetNeighbourSquareScore(*self.snake.body[1]) != 0):
                return -100000
            if(self.GetNeighbourValues(*self.snake.body[1])).count(0) >= 1:
                # These means in our previous square we had another possible move
                return -10000 + self.score
            return self.score

        f2 = defaultSnakeSize

        if pow(self.score, f2) < int(pow(len(self.snake.body), f2)):
            self.score = int(pow(len(self.snake.body), f2))
        
        return self.score 

# If fruit is next give 1 else give 0
    def GetNeighbourSquareScore(self, x, y):
        moves = self.GetMoves()
        for move in moves:
            moveX, moveY = move
            if self.board[moveX][moveY] == "🍎":
                return GetDirection(x, y, moveX, moveY)
        return 0
    
    def GetNeighbourValues(self, x, y):
        list = []
        values = []

        if x > 0:
            list.append((x-1, y))
        else:
            list.append(None)
        if x < self.size - 1:
            list.append((x+1, y))
        else:
            list.append(None)
        if y > 0:
            list.append((x, y-1))
        else:
            list.append(None)
        if y < self.size - 1:
            list.append((x, y+1))
        else:
            list.append(None)
        
        for i in range(4):
            if list[i] is not None:
                value = self.board[list[i][0]][list[i][1]]
                if value == "🍎":
                    values.append(1)
                elif value == "🐍":
                    values.append(2)
                else:
                    values.append(0)
            else:
                values.append(-1)
        return values
                


    def GetState(self):
        x, y = self.snake.body[0]
        features = [self.snake.Direction(), self.snake.DistanceToFood(x, y, self.fruit[0], self.fruit[1]), len(self.snake.body), self.moveCount]

        scalerX = 0
        scalerY = 0
        for i in self.snake.body:
            scalerX += i[0]
            scalerY += i[1]

        scalerX = scalerX / len(self.snake.body)
        scalerY = scalerY / len(self.snake.body)
        # We do this so we know which corner the snake is in
        features.extend([scalerX, scalerY])
        features.extend([x, y])
        features.extend([self.fruit[0], self.fruit[1]])
        features.extend(self.GetNeighbourValues(x, y))
        return features

def isNeighborXY(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2) == 1
        
class Snake:
    def __init__(self, snake_size): 
        self.body = self.MakeBaseBody(snake_size)

    def MakeBaseBody(self, board_size): 
        head =  board_size//2, board_size//2
        body = Queue()
        body.put(head)
        body.put((head[0], head[1] - 1))
        body.put((head[0], head[1] - 2))
        return body.queue
    
    def Move(self, x,y, board):
        if board[x][y] != "🍎":   
            self.body.pop()
        self.body.insert(0, (x,y))
        self.UpdateDirection()
        if board[x][y] == "🍎":
            return "Fruit"
        
    def UpdateDirection(self):
        self.direction = self.Direction()

    def Direction(self):
        x1, y1 = self.body[0]
        x2, y2 = self.body[1]
        # 1 up, 2 down, 3 left, 4 right
        if x1 == x2:
            if y1 > y2:
                return 3
            else:
                return 4
        else:
            if x1 > x2:
                return 1
            else:
                return 2
            
    def getNearestWallDistance(self, x, y, board_size):
        return min(x, y, board_size - x, board_size - y)
    
    def HeadToTailDistance(self):
        x1, y1 = self.body[0]
        x2, y2 = self.body[-1]
        return abs(x1 - x2) + abs(y1 - y2)
        
    def DistanceToFood(self, x, y, fx, fy):
        return abs(x - fx) + abs(y - fy)
    

    
            

