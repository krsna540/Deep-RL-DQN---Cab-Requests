import numpy as np
import math
import random
import itertools
from itertools import product

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.locations = np.arange(0,5) # Taking each location as a integer 0 is for A , 1 is for B and so on
        self.hours = np.arange(0, 24,dtype=np.int) # Hours are encoded as 24
        self.days = np.arange(0, 7) # days of week encoded as 0 to 6

        self.action_space = [a for a in list(product(self.locations, self.locations)) if a[0] != a[1]]  # 20 possible actions
        self.action_size = len(self.action_space)
        # STATE is [location, hours, days]
        self.state_space = [a for a in list(product(self.locations, self.hours, self.days))]
        self.state_init =self.state_space[np.random.randint(len(self.state_space))]

        self.tempcommutetime = 0 # placeholder for commute time when pickup & drop locations are not same
        self.SingleRideTime = 0 # time spent in dropping passenger for single t
        self.RideTime = 0 # total commute time for pickup and drop. RideTime=tempcommutetime+SingleRideTime
        self.state_size=36 # for Algo1 state size will be m+t+d i.e 5+24+7
        #self.state_size=46 #for Algo2 state size will be m + t + d + m + m i.e 5+24+7+5+5=4

        self.reset()

    # Encoding state (or state-action) for NN input for architecture 1
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN.This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        locVector = np.eye(m, dtype=np.int16)[int(state[0])]
        hourVector = np.eye(t, dtype=np.int16)[int(state[1])]
        dayVector = np.eye(d, dtype=np.int16)[int(state[2])]
        state_encod = np.concatenate((locVector, hourVector, dayVector))
        return state_encod

# Encoding state (or state-action) for NN input for architecture 2

    def state_encod_arch2(self,state,action):
        """convert the state into a vector so that it can be fed to the NN.This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        locVector = np.eye(m, dtype=np.int16)[int(state[0])]
        hourVector = np.eye(t, dtype=np.int16)[int(state[1])]
        dayVector = np.eye(d, dtype=np.int16)[int(state[2])]
        pickupVector = np.eye(m, dtype=np.int16)[action[0]]
        dropVector = np.eye(m, dtype=np.int16)[action[1]]
        state_encod = np.concatenate((locVector, hourVector, dayVector, pickupVector, dropVector))
        return state_encod

        # Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        loc_index = state[0]
        if loc_index == 0 :
            requests = np.random.poisson(2)
        elif loc_index == 1 :
            requests = np.random.poisson(12)
        elif loc_index == 2 :
            requests = np.random.poisson(4)
        elif loc_index == 3 :
            requests = np.random.poisson(7)
        elif loc_index == 4 :
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        possibleActions = random.sample(range(0, (m-1)*m), requests)
        actions = [self.action_space[i] for i in possibleActions]
        actions.append((0, 0))  # adding the drive going to offline condition as (0,0) represents offline condition
        return possibleActions, actions


    def reward_func(self, state, action, Time_matrix,flag):
        """Takes in state, action and Time-matrix and returns the reward"""
        # Reward function returns revenue earned from pickup point p to drop point q)
        """𝑅(𝑠=𝑋𝑖𝑇𝑗𝐷𝑘) ={ 𝑅𝑘∗(𝑇𝑖𝑚𝑒(𝑝,𝑞)) − 𝐶𝑓 ∗(𝑇𝑖𝑚𝑒(𝑝,𝑞) + 𝑇𝑖𝑚𝑒(𝑖,𝑝))𝑎=(𝑝,𝑞)
                        −𝐶𝑓       𝑎=(0,0)}"""
        if flag==1:
            reward=(self.SingleRideTime*R)-(5*(self.tempcommutetime*+self.SingleRideTime))
        else:
            reward=-self.tempcommutetime
        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""

        if not isinstance(action,tuple):
            action=self.action_space[action]
        is_terminal=False
        self.tempcommutetime=0
        self.SingleRideTime=0

        currLocation=state[0]
        totalHours=state[1]
        totaldays=state[2]

        pickupLocation=action[0]
        dropLocation=action[1]

        if action[0]!=0 and action[1]!=0:
            if currLocation!=pickupLocation:
                self.tempcommutetime=Time_matrix[currLocation][pickupLocation][int(totalHours)][totaldays]
                totalHours=self.tempcommutetime+totalHours

                totalHours,totaldays= self.DayModifier(totalHours,totaldays)
                currLocation=pickupLocation

            self.SingleRideTime=Time_matrix[currLocation][dropLocation][int(totalHours)][totaldays]

            totalHours=totalHours+self.SingleRideTime
            totalHours,totaldays= self.DayModifier(totalHours,totaldays)
            self.RideTime=self.RideTime+self.tempcommutetime+self.SingleRideTime
            #flag as 1 to indicate successful trip instead of offline
            reward=self.reward_func(state,action,Time_matrix,1) # ride completion reward
        else:
            dropLocation=currLocation
            totalHours,totaldays=self.DayModifier(totalHours+1,totaldays)
            #flag as 0 to indicate offline trip
            reward=self.reward_func(state,action,Time_matrix,0)  #Offline reward
            # 24*30 days=720, end of the month
        if self.RideTime>=720:
            is_terminal=True
        next_state=(dropLocation,totalHours,totaldays) #state returned without encoding
        return next_state,action,reward,is_terminal


    def DayModifier(self, hour, nextday):
        """Time and week day modifier, Handling changing the week day based on time """
        while hour >= 24:
            if hour == 24:
                nextday = nextday+1
                hour = 0
            elif hour > 24:
                nextday = nextday+1
                hour = hour-24
            if nextday > 6:
                nextday = nextday-7
        return (hour, nextday)



    def reset(self):
        'Reseting the object and setting total commute as zero'
        self.RideTime=0
        return self.action_space, self.state_space, self.state_init
