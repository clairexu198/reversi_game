'''
Created on Feb 20, 2021

@author: Claire Xu
'''
import random, sys, pygame, time, copy,math
import numpy as np




from pygame.locals import *
from keras.models import Model,load_model,save_model
from keras.layers import Conv2D, Dense, Flatten, Input


# constants

FPS = 10 # frames per second to update the screen
WINDOWWIDTH = 640 # width of the program's winsdow, in pixels
WINDOWHEIGHT = 480 # height in pixels
SPACESIZE = 50 # width & height of each space on the board, in pixels
BOARDWIDTH = 8 # how many columns of spaces on the game board
BOARDHEIGHT = 8 # how many rows of spaces on the game board
WHITE_TILE = -1 # an arbitrary but unique value
BLACK_TILE = 1 # an arbitrary but unique value
EMPTY_SPACE = 0 # an arbitrary but unique value
HINT_TILE = 'HINT_TILE' # an arbitrary but unique value
ANIMATIONSPEED = 25 # integer from 1 to 100, higher is faster animation


# Amount of space on the left & right side (XMARGIN) or above and below
# (YMARGIN) the game board, in pixels.
XMARGIN = int((WINDOWWIDTH - (BOARDWIDTH * SPACESIZE)) / 2)
YMARGIN = int((WINDOWHEIGHT - (BOARDHEIGHT * SPACESIZE)) / 2)

#              R    G    B
WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
GREEN      = (  0, 155,   0)
BRIGHTBLUE = (  0,  50, 255)
BROWN      = (174,  94,   0)

TEXTBGCOLOR1 = BRIGHTBLUE
TEXTBGCOLOR2 = GREEN
GRIDLINECOLOR = BLACK
TEXTCOLOR = WHITE
HINTCOLOR = BROWN

time1=0.0
time2=0.0
time3=0.0





# class to store board state - later to be passed to networks/agents
class BState:
    def __init__(self,board,thisturn,previous):
        self.board=np.copy(board)  # board is a numpy array of 8x8
        self.thisturn=thisturn   # current turn, 1 for black, -1 for white
        self.previous = previous # a tuple, coordinates for the previous move, (-1,-1) for pass
        self.scores = []  # list to store how many tiles to be turned for each move
        self.moves=self.getValidMoves()



    def getMask(self):
        mask =  np.zeros((BOARDWIDTH,BOARDHEIGHT),dtype=int)   # a mask showing where next moves can be played
        for move in self.moves:
            mask[move[0]][move[1]] = 1  
        return mask
            
       
    def getScore(self):
        black=np.count_nonzero(self.board == BLACK_TILE)
        white=np.count_nonzero(self.board == WHITE_TILE)
        return (black,white)
    
 
    def isValidMove(self,xstart, ystart):
        # Returns False if the player's move is invalid. If it is a valid
        # move, returns a list of spaces of the captured pieces.
        if self.board[xstart][ystart] != 0 or not self.isOnBoard(xstart, ystart):
            return False
    
        tile=self.thisturn  # the color to be played
        self.board[xstart][ystart] = tile  # temporarily fill the position with the tile
    
        otherTile=-tile
        
        
        tilesToFlip = []
        # check each of the eight directions:
        for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = xstart, ystart
            x += xdirection
            y += ydirection
            if self.isOnBoard(x, y) and self.board[x][y] == otherTile:
                # The piece belongs to the other player next to our piece.
                x += xdirection
                y += ydirection
                if not self.isOnBoard(x, y):  # reached the edge
                    continue
                while self.board[x][y] == otherTile:
                    x += xdirection
                    y += ydirection
                    if not self.isOnBoard(x, y):  # edge of the board
                        break # break out of while loop, continue in for loop
                if not self.isOnBoard(x, y):
                    continue
                if self.board[x][y] == tile:
                    # There are pieces to flip over. Go in the reverse
                    # direction until we reach the original space, noting all
                    # the tiles along the way.
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        tilesToFlip.append([x, y])
    
        self.board[xstart][ystart] = EMPTY_SPACE # make space empty
        if len(tilesToFlip) == 0: # If no tiles flipped, this move is invalid
            return False
        return tilesToFlip
    
    
    def isOnBoard(self,x, y):
        # Returns True if the coordinates are located on the board.
        return x >= 0 and x < BOARDWIDTH and y >= 0 and y < BOARDHEIGHT
    
    
    
    
    def getValidMoves(self):
        # Returns a list of (x,y) tuples of all valid moves.
        validMoves = []
    
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                if self.board[x][y] !=0:  # already played
                    continue
                toturn=self.isValidMove(x, y)
                if  toturn != False:
                    validMoves.append((x, y))
                    self.scores.append(len(toturn))
        return validMoves
       
       
    def makeMove(self,mymove): #make a move
        x=mymove[0]
        y=mymove[1]
        if not(x == -1 and y == -1): #not a pass

        
        
            tilestoturn=self.isValidMove(x, y)
            if tilestoturn == False: 
                return False
            
            
            
            tile=self.thisturn
            self.board[x][y] = tile
            
            for turn in tilestoturn:   # tiles to be turned
                self.board[turn[0]][turn[1]] = tile 
        else: # a pass
            if len(self.moves) != 0:
                return False   # valid moves exist, cannot pass
        
        self.previous=(x,y)   # this is true even if the move isa pass
        self.thisturn = -self.thisturn
        
        

        self.scores = []  # list to store how many tiles to be turned for each move
        #now update valid moves
        self.moves=self.getValidMoves()


        return True
            
            
            
    def is_Over(self):
        if np.count_nonzero(self.board == EMPTY_SPACE) == 0:
            return True
        
        return (self.previous == (-1,-1) and len(self.moves) == 0)
    
    
class PlayerAgent():    # random agent
    def select_move(self,state):
        moves=state.moves
        if len(moves) == 0:
            return (-1,-1)    # a pass
        return random.choice(moves)

class RandomAgent():    # random agent
    def select_move(self,state):
        moves=state.moves
        if len(moves) == 0:
            return (-1,-1)    # a pass
        return random.choice(moves)
    

class SimpleAgent():   # agent based on score of one move, plus prioritizing playing corners
    def select_move(self,state):
        moves=state.moves
        if len(moves) == 0:
            return (-1,-1)    # a pass
   
        move_score=list(zip(moves,state.scores))
        random.shuffle(move_score)
        
        bestscore=0
        mymove=(-1,-1)
        for tmp in move_score:
            pos=tmp[0]
            score=tmp[1]
            if pos == (0,0) or pos == (0,7) or pos == (7,7) or pos == (7,0):  # corner?
                return pos
            
            if score > bestscore:
                bestscore = score
                mymove=pos
                
        return mymove





###
# here is the MCTS part


class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = BState(game_state.board,game_state.thisturn,game_state.previous)
        self.parent = parent
        self.move = move
        self.win_counts = [0,0]  #black and white win counts
        self.num_rollouts = 0
        self.children = []
        unvisited_moves = []+self.game_state.moves  # make a copy, do not change state.moves
        if len(unvisited_moves) == 0: # pass
            unvisited_moves = [(-1,-1)]
        self.unvisited_moves = unvisited_moves

    
    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        
        new_game_state = BState(self.game_state.board,self.game_state.thisturn,self.game_state.previous)
        new_game_state.makeMove(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node
    
    def record_win(self, winner):
        if winner >=0 :   # -1 is a draw
            self.win_counts[winner] += 1
        self.num_rollouts += 1
        
        
        
    def can_add_child(self):
        return len(self.unvisited_moves) > 0
    
    def is_terminal(self):
        return self.game_state.is_Over()
    
    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)
    
    
    
    
def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
    exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_pct + temperature * exploration

class MCTSAgent():    # Monte Carlo Tree Search Agent
    def __init__(self,num_rounds,temperature,agent):
        self.num_rounds = num_rounds
        self.agent = agent
        self.temperature = temperature

    def simulate_game(self,state0):  #simulate a game from state0, using agent, and return the winner
#        global time1,time2,time3
        
        state=BState(state0.board,state0.thisturn,state0.previous)  #make a copy of current state
        
        
        while True:
#            t0=time.time()
            if state.is_Over(): 
                break
#            t1=time.time()
            move = self.agent.select_move(state)
#            t2=time.time()
            state.makeMove(move)
#            t3=time.time()
#            time1+=t1-t0
#            time2+=t2-t1
#            time3+=t3-t2            
            
        
        
        scores = state.getScore()
        
        
        
        

        if scores[0] > scores[1]: 
            #black wins
            return 0
        elif scores[0] < scores[1]:
            #white wins
            return 1
        else:
            #draw
            return -1
        
    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        #print(total_rollouts,node.num_rollouts)
        best_score = -1.0
        best_child = None
        players=[1,-1,0]
        next_player = players[node.game_state.thisturn+1]
        
        for child in node.children:            
            score = uct_score(total_rollouts,child.num_rollouts, child.winning_frac(next_player), self.temperature)
            if score > best_score :
                best_score = score
                best_child = child
        return best_child
        
    def select_move(self,state):
        # global time1,time2,time3
        #
        # time1=0
        # time2=0
        # time3=0
        
        root = MCTSNode(state)
        for i in range(self.num_rounds):
            node = root
            

            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)
            
            
  
            
            
            if node.can_add_child(): 
                node = node.add_random_child() 
                
               

            winner = self.simulate_game(node.game_state) 


            while node is not None: 
                node.record_win(winner) 
                node = node.parent
                

        
        # now the tree is finished (num_rounds), select move
        best_move = None
        best_pct = -1.0
        players=[1,-1,0]
        for child in root.children:
            next_player=players[state.thisturn+1]  # black or white (i.e. 0 or 1) 
        
            child_pct = child.winning_frac(next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
                
                
        #print(time1,time2,time3)

        return best_move









####
# here is the deep neural network
#

class SimpleEncoder():  # a three plane encoder for Reversi board
    def __init__(self):
        self.width=8
        self.height=8
        self.num_planes=4
        
    def name(self):
        return 'fourplane'
    
    def encode(self,state):
        board_tensor=np.zeros((self.num_planes,self.height,self.width),dtype=int)
        
        #black
        plane=np.zeros((self.height,self.width),dtype=int)
        plane[state.board > 0] = 1
        board_tensor[0]=plane
        
        #white
        plane=np.zeros((self.height,self.width),dtype=int)
        plane[state.board < 0] = -1
        board_tensor[1]=plane
        
        #possible moves (mask)
        plane=state.getMask()*state.thisturn
        board_tensor[2]=plane
        
        
        #plane indicating which player to play
        plane=np.zeros((self.height,self.width),dtype=int)
        plane[plane == 0] = state.thisturn
        board_tensor[3]=plane
        
        return board_tensor
    
    



# a simple model, povides oolicy
    
class PolicyAgent():
    def __init__(self):
        board_input = Input(shape=(4,8,8), name='board_input')
        conv1 = Conv2D(64, (3, 3),  padding='same',  activation='relu')(board_input) 
        conv2 = Conv2D(64, (3, 3), padding='same',activation='relu')(conv1) 
        conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2) 
        flat = Flatten()(conv3)
        processed_board = Dense(512)(flat) 
        policy_hidden_layer = Dense(512, activation='relu')(processed_board) 
        policy_output = Dense(64, activation='softmax')(policy_hidden_layer) 
        self.model = Model(inputs=board_input,outputs=[policy_output])

    
# another policy model with larger network
class PolicyAgent2():
    def __init__(self):
        board_input = Input(shape=(4,8,8), name='board_input')
        conv1 = Conv2D(64, (3, 3),  padding='same',  activation='relu')(board_input) 
        conv2 = Conv2D(64, (3, 3), padding='same',activation='relu')(conv1) 
        conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2) 
        flat = Flatten()(conv3)
        processed_board = Dense(2048)(flat) 
        policy_hidden_layer = Dense(512, activation='relu')(processed_board) 
        policy_output = Dense(64, activation='softmax')(policy_hidden_layer) 
        self.model = Model(inputs=board_input,outputs=[policy_output])

    
    

# the model that provides both policy and value
    
class ValueAgent():
    def __init__(self):
        board_input = Input(shape=(4,8,8), name='board_input')
        conv1 = Conv2D(64, (3, 3),  padding='same',  activation='relu')(board_input) 
        conv2 = Conv2D(64, (3, 3), padding='same',activation='relu')(conv1) 
        conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2) 
        flat = Flatten()(conv3)
        processed_board = Dense(512)(flat) 
        policy_hidden_layer = Dense(512, activation='relu')(processed_board) 
        policy_output = Dense(64, activation='softmax')(policy_hidden_layer) 
        value_hidden_layer = Dense(512, activation='relu')(processed_board) 
        value_output = Dense(1, activation='tanh')(value_hidden_layer)
        self.model = Model(inputs=board_input,outputs=[policy_output, value_output])



class SimpleValueAgent():
    def __init__(self):
        board_input = Input(shape=(4,8,8), name='board_input')
        conv1 = Conv2D(64, (3, 3),  padding='same',  activation='relu')(board_input) 
        conv2 = Conv2D(64, (3, 3), padding='same',activation='relu')(conv1) 
        #conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2) 
        flat = Flatten()(conv2)
        processed_board = Dense(512)(flat) 
        policy_hidden_layer = Dense(512, activation='relu')(processed_board) 
        policy_output = Dense(64, activation='softmax')(policy_hidden_layer) 
        value_hidden_layer = Dense(512, activation='relu')(processed_board) 
        value_output = Dense(1, activation='tanh')(value_hidden_layer)
        self.model = Model(inputs=board_input,outputs=[policy_output, value_output])

        
    
    
# using the value model, take the best policy prediction
class MyAgent():
    def __init__(self,model,encoder):
        self.model = model
        self.encoder = encoder
        
    def saveModel(self,modelpath):
        
        #usage:
        # with h5py.File(output_file, 'w') as outf:
            # agent.serialize(outf)
        self.model.save(modelpath)
                
    
    def predict(self,state):
        board_tensor=self.encoder.encode(state)
        return self.model.predict(np.array([board_tensor]))

    def select_move(self,state):
        board_tensor=self.encoder.encode(state)
        mask=state.getMask()
        
        x=self.model.predict(np.array([board_tensor]))
        if isinstance(x,tuple):  # in case of returning value include both policy and value
            x=x[0]
        policy=np.reshape(x[0],(8,8))
        policy+=0.001  # make sure there is no zeros - this will prevent errors of possibly selecting non-playable positions
        policy=policy*mask
        if np.count_nonzero(policy) == 0:
            return (-1,-1)
        
        pmax=np.max(policy)
        
        result=np.where(policy == pmax)
        
        move=list(zip(result[0],result[1]))
        
        return move[0]




# using the value model, but randomly (with probability distribution) takes policy prediction

# used for reinforcement learning


class RLAgent():   # agent for re-enforcement learning - randomize selections
    def __init__(self,model,encoder):
        self.model = model
        self.encoder = encoder
        
    def saveModel(self,modelpath):
        
        #usage:
        # with h5py.File(output_file, 'w') as outf:
            # agent.serialize(outf)
        self.model.save(modelpath)
                
    
    def predict(self,state):
        board_tensor=self.encoder.encode(state)
        return self.model.predict(np.array([board_tensor]))

    def select_move(self,state,value=[0]):
        board_tensor=self.encoder.encode(state)
        mask=state.getMask()
        
        x,y=self.model.predict(np.array([board_tensor]))
        
        if len(value) == 0: # if the user supplies a empty list, then return the value
            value.append(y[0][0])
        
        mask=np.reshape(mask,64)  # reshape to 1-D vector

        
        policy=x[0]+0.001  # make sure there is no zeros - this will prevent errors of possibly selecting non-playable positions
        policy=policy*mask
        if np.count_nonzero(policy) == 0:  # pass
            return (-1,-1)
        
        # now normalize policy so that probabilities add up to 1
        policy=policy/np.sum(policy)
        
        result=np.random.choice(64,p=policy)
        
        row = result // 8
        column = result % 8
        
        return (row,column)





# Here is the part based on Alpha Zero (Zeronodes and ZeroAgent)

class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        
        
class ZeroTreeNode:

    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent                      # <1>
        self.last_move = last_move                # <1>
        self.total_visit_count = 1
        self.branches = {}
        if np.count_nonzero(priors) == 0: #pass
            self.branches[(-1,-1)]=Branch(1.0)
        else:
            for i in range(64):
                if priors[i] == 0:
                    continue
                row=i//8
                column=i%8
                self.branches[(row,column)]=Branch(priors[i])
                
                
        self.children = {}                        # <2>

    def moves(self):                              # <3>
        return self.branches.keys()               # <3>

    def add_child(self, move, child_node):        # <4>
        self.children[move] = child_node          # <4>

    def has_child(self, move):                    # <5>
        return move in self.children              # <5>

    def get_child(self, move):                    # <6>
        return self.children[move]                # <6>
# end::node_class_body[]

# tag::node_record_visit[]
    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value
# end::node_record_visit[]

# tag::node_class_helpers[]
    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0



class ZeroAgent():
# end::zero_defn[]
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder

        self.collector = None

        self.num_rounds = rounds_per_move
        self.c = c

# tag::zero_select_move_defn[]
    def select_move(self, game_state,value=[],vc_return=[]):
# end::zero_select_move_defn[]
# tag::zero_walk_down[] 
        
        if len(game_state.moves) == 0:  #pass
            return (-1,-1)
        if len(game_state.moves) ==1:  # only one move, no need to waste time
            if len(vc_return) ==64:
                move=game_state.moves[0]
                row=move[0]
                column=move[1]
                vc_return[row*8+column]=self.num_rounds

            return game_state.moves[0]
        
        root = self.create_node(game_state)           # <1>

        for i in range(self.num_rounds):              # <2>
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):          # <3>
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
# end::zero_walk_down[]

# tag::zero_back_up[]
            #make a copy of old state to new state
            new_state=BState(node.state.board,node.state.thisturn,node.state.previous)
            new_state.makeMove(next_move)
            child_node = self.create_node(
                new_state, move=next_move, parent=node)

            move = next_move
            value0 = -1 * child_node.value             # <1>
            while node is not None:
                node.record_visit(move, value0)
                move = node.last_move
                node = node.parent
                value0 = -1 * value0
# end::zero_back_up[]

        # return visit count if needed
        if len(vc_return) ==64:
            
            for i in range(64):
                row=i//8
                column=i%8
                move=(row,column)
                vc_return[i]==root.visit_count(move)


        
        
        # for move in root.moves():
            # print(move,root.branches[move].prior,root.visit_count(move))
# tag::zero_select_max_visit_count[]
        return max(root.moves(), key=root.visit_count)


# tag::zero_select_branch[]
    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return max(node.moves(), key=score_branch)             # <1>
# end::zero_select_branch[]

# tag::zero_create_node[]
    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])                 # <1>
        priors, values = self.model.predict(model_input)
        priors = priors[0]                                     # <2>
        value = values[0][0]
        mask=game_state.getMask()
        mask=np.reshape(mask,64)
        
        priors=priors*mask
        
        if np.count_nonzero(priors) != 0: 
            #not a pass - normalize
            priors=priors/np.sum(priors)
        
       
        new_node = ZeroTreeNode(
            game_state, value,
            priors,
            parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
# end::zero_create_node[]

# tag::zero_train[]
    # def train(self, experience, learning_rate, batch_size):     # <1>
        # num_examples = experience.states.shape[0]
        #
        # model_input = experience.states
        #
        # visit_sums = np.sum(                                    # <2>
            # experience.visit_counts, axis=1).reshape(           # <2>
            # (num_examples, 1))                                  # <2>
        # action_target = experience.visit_counts / visit_sums    # <2>
        #
        # value_target = experience.rewards
        #
        # self.model.compile(optimizer=SGD(lr=learning_rate),
            # loss=['categorical_crossentropy', 'mse'])
        # self.model.fit(
            # model_input, [action_target, value_target],
            # batch_size=batch_size)
# # end::zero_train[]





# a match engine to run a match with two agents
class match():
    def __init__(self,black_agent,white_agent):
        self.black_agent=black_agent
        self.white_agent=white_agent
        
        
    def runMatch(self):   # returns two various: win and scores
        board=np.zeros((8,8),dtype=int)
        board[3][3]=1
        board[4][4]=1
        board[3][4]=-1
        board[4][3]=-1
        
        state=BState(board,1,(-1,-1))
        
        
        # now play
        agent1=self.black_agent
        agent2=self.white_agent
        
        while True:
                        
            if state.is_Over(): # 2 passes -> game ends
                break
            move=agent1.select_move(state)
            state.makeMove(move)

    
            
            if state.is_Over(): # 2 passes -> game ends
                break
            move=agent2.select_move(state)
            state.makeMove(move)
           
        scores = state.getScore()
        
        if scores[0] > scores[1]: 
            #black wins
            return 0,scores
        elif scores[0] < scores[1]:
            #white wins
            return 1,scores
        else:
            #draw
            return -1,scores
                    



    def debugMatch(self):   # returns two various: win and scores
        board=np.zeros((8,8),dtype=int)
        board[3][3]=1
        board[4][4]=1
        board[3][4]=-1
        board[4][3]=-1
        
        state=BState(board,1,(-1,-1))
        
        
        # now play
        agent1=self.black_agent
        agent2=self.white_agent
        
        while True:
                        
            if state.is_Over(): # 2 passes -> game ends
                break
            
            value=[]
            move=agent1.select_move(state,value=value)
            print(state.board,value[0],state.thisturn,move)
            state.makeMove(move)

    
            
            if state.is_Over(): # 2 passes -> game ends
                break
            value=[]
            move=agent2.select_move(state,value=value)
            print(state.board,value[0],state.thisturn,move)
            state.makeMove(move)
           
        scores = state.getScore()
        
        if scores[0] > scores[1]: 
            #black wins
            return 0,scores
        elif scores[0] < scores[1]:
            #white wins
            return 1,scores
        else:
            #draw
            return -1,scores
        
        
        
    def stepMatch(self):   # step by step, for debugging
        board=np.zeros((8,8),dtype=int)
        board[3][3]=1
        board[4][4]=1
        board[3][4]=-1
        board[4][3]=-1
        
        state=BState(board,1,(-1,-1))
        
        # myenc = SimpleEncoder()
        # deepagent=MyAgent()
        
        # now play
        agent1=self.black_agent
        agent2=self.white_agent
        
        while True:
            if state.is_Over(): # 2 passes -> game ends
                break
            move=agent1.select_move(state)
            state.makeMove(move)
            print("Black move: ",move)
            print(state.board)
    
            
            if state.is_Over(): # 2 passes -> game ends
                break
            move=agent2.select_move(state)
            state.makeMove(move)
            print("White move: ",move)
            
            print(state.board)
            
            # board_tensor=myenc.encode(state)
            # print(deepagent.model.predict(np.array([board_tensor])))
    
 
           
        scores = state.getScore()
        
        if scores[0] > scores[1]: 
            #black wins
            return 0,scores
        elif scores[0] < scores[1]:
            #white wins
            return 1,scores
        else:
            #draw
            return -1,scores
              
            
            
     
                 

