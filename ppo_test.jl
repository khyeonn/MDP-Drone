if !@isdefined Drone
    include("Drone.jl")
    include("ppo.jl")
end
using .Drone: DroneEnv, DroneAction, plot_frame, reset!, gen, create_animation, isterminal
using .ppo: PPO, Actor, Critic, get_action, ppo_test
using Statistics: mean
using Plots
using BSON: @load

@load "models/ppo_final.bson" ppo_network


# test actor agent
n_episodes = 1_000
rewards, ep_lengths, batch_env, batch_acts, dones = ppo_test(ppo_network, n_episodes=n_episodes)
println("Average rewards over $n_episodes is $(mean(rewards))")
println("Number of sucesses in $n_episodes is $dones")

# plot learning curve
p = plot(1:length(rewards), rewards, xlabel="Episodes", ylabel="Total rewards", title="Learning curve")
display(p)

# get best performing simulation
i = argmax(rewards)
animation = create_animation(batch_env[i])
gif(animation, "gif_test.gif")