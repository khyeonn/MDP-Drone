if !@isdefined Drone
    include("Drone.jl")
end
using POMDPs
using .Drone: DroneEnv, DroneState, DroneAction, isterminal, discount, gen, render

env = DroneEnv()

target = (10.0, 10.0, 5.0) # target region (x, y, radius)
initial_state = DroneState(0.0, 0.0, 0.0, 0.0) # state is (x, y, v, theta)

action = DroneAction(0.5, 0.5) # actions are (accelerate, turn). turn action is in radians

# can transition forward to next state and get reward like this:
new_state = POMDPs.transition(env, initial_state, action)
reward = POMDPs.reward(env, new_state) 

# can also do this in one line:
sp, r = gen(env, initial_state, action) 

# render environment like this:
render(env, sp)