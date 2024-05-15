if !@isdefined Drone
    include("Drone.jl")
    include("ppo.jl")
end
using .Drone: DroneEnv, DroneAction, plot_frame, reset!, gen, create_animation, isterminal
using .ppo: PPO, Actor, Critic, get_action, ppo_test
using Statistics: mean
using Plots
using Random
using BSON: @load

@load "models/ppo_final_v2.bson" ppo_network

# rng = MersenneTwister(12712)
rng = MersenneTwister(2)
# rng=nothing

# test actor agent
n_episodes = 5_000
rewards, ep_lengths, batch_env, batch_acts, dones, h_rewards, h_lengths, h_envs, h_dones = ppo_test(ppo_network, n_episodes=n_episodes, rng=rng)

# plot learning curve
p = plot(1:length(rewards), rewards, xlabel="Episodes", ylabel="Total rewards", label="PPO", legend=true, title="Learning Curve")
plot!(1:length(h_rewards), h_rewards, xlabel="Episodes", ylabel="Total rewards", label="Heuristic", legend=true)
display(p)
savefig("figures/learning_curve.png")

# get best performing simulation
i = argmax(rewards)
animation = create_animation(batch_env[i])
g = gif(animation, "test_ppo.gif")
display(g)

# create animation
i = argmax(h_rewards)
animation = create_animation(h_envs[i])
g = gif(animation, "test_heuristic.gif")
display(g)

println("------PPO performance:------")
println("Average rewards over $n_episodes is $(mean(rewards))")
println("Number of sucesses in $n_episodes is $dones")
println("------Heuristic performance:------")
println("Average rewards over $n_episodes is $(mean(h_rewards))")
println("Number of sucesses in $n_episodes is $h_dones")
