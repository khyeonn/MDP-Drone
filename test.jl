if !@isdefined Drone
    include("Drone.jl")
end
using POMDPs
using Plots
using .Drone: DroneEnv, DroneState, DroneAction, Target, Obstacle, isterminal, discount, gen, plot_frame, set_target!, reset!, create_animation

# env generates random target region
env = DroneEnv()

# render using the drone state
plot_frame(env)

### can override auto generated target like this:
# target = Target(10.0, 10.0, 5.0) # target region (x, y, radius)
# set_target!(env, target)

a = DroneAction(0.5, 0.5) # actions are (accelerate, turn). turn action is in radians

# can transition forward to next state and get reward like this:
new_state, done = POMDPs.transition(env, a)
reward = POMDPs.reward(env) 

# can also do this in one line:
sp, r, done = gen(env, a) 

# render environment:
plot_frame(env)

# reset env like this:
reset!(env)


#### EXAMPLE ####
env = DroneEnv()

# store environment for animating:
batch_env = Vector{DroneEnv}()

max_t = 50
for _ in 1:max_t
    # choose random actions
    action = DroneAction(randn(), randn())

    # forward step env
    sp_, r_, done_ = gen(env, action)

    # add env to batch
    push!(batch_env, deepcopy(env))

    if done_
        break
    end
end

# create animation
animation = create_animation(batch_env)
gif(animation, "gif_ex.gif")