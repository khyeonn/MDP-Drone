if !@isdefined Drone
    include("DroneDiscrete.jl")
end
using POMDPs
using .DroneDiscrete: gen, DroneEnv, DroneState, render

env = DroneEnv()

initial_state = DroneState(1, 1, 0.0) # state is (x, y, v, theta)

action = :FWD

# # can transition forward to next state and get reward like this:
# new_state = POMDPs.transition(env, initial_state, action)

# reward = POMDPs.reward(env, rand(new_state), action) 

# can also do this in one line:
sp, r = gen(env, initial_state, action) 

# render environment like this:
render(env, rand(sp))