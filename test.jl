if !@isdefined Drone
    include("Drone.jl")
end
using POMDPs
using .Drone

m = DroneEnv()

target = (10.0, 10.0, 5.0) # target region (x, y, radius)
initial_state = DroneState(0.0, 0.0, 0.0, 0.0) # state is (x, y, v, theta)

action = DroneAction(0.5, 0.1) # actions are (accelerate, turn). turn action is in radians

new_state = POMDPs.transition(m, initial_state, action)

reward = POMDPs.reward(new_state, target) 