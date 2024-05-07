if !@isdefined Drone
    include("Drone.jl")
end
using POMDPs
using .Drone: DroneEnv, DroneState, DroneAction, Target, Obstacle, isterminal, discount, gen, render, set_target!

# env generates random target region
env = DroneEnv()

# render using the drone state
render(env, env.drone)

### can override auto generated target like this:
# target = Target(10.0, 10.0, 5.0) # target region (x, y, radius)
# set_target!(env, target)

action = DroneAction(0.5, 0.5) # actions are (accelerate, turn). turn action is in radians

# can transition forward to next state and get reward like this:
new_state = POMDPs.transition(env, action)
reward = POMDPs.reward(env) 

# can also do this in one line:
sp, r = gen(env, action) 

# render environment:
render(env, sp)