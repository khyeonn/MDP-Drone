if !@isdefined Drone
    include("Drone.jl")
    include("ppo.jl")
end
using POMDPs
using Plots
using .Drone: DroneEnv, plot_frame
using .ppo: PPO, Actor, Critic, learn


function train(env, actor, critic; hyperparameters)
    model = PPO(env, actor, critic)

    if !isempty(hyperparameters)
        ppo._init_hyperparameters(model, hyperparameters)
    end

    actor_loss, critic_loss, rewards = learn(model)

    return actor_loss, critic_loss, rewards
end


function plot_loss(data, label::String)
    p = plot(1:length(data), data, label=label, xlabel="Time steps", ylabel="Loss")
    display(p)
end

function plot_rewards(reward_vector)
    for i in eachindex(reward_vector)
        j = i*10
        p = plot(1:length(reward_vector[i]), reward_vector[i], label="Reward for Iteration $j", xlabel="Time steps", ylabel="Rewards")
        display(p)
    end
end

hyperparameters = Dict(
        "batch_size" => 5_000,
        "mini_batch_size" => 200,
        "max_timesteps_per_episode" => 5_000,
        "updates_per_iteration" => 5,
        "total_timesteps" => 60_000_000,
        "soft_update_coeff" => 0.005,
        "entropy_coeff" => 0.001,
        "lr" => 1e-4,
        "clip" => 0.2
    )

env = DroneEnv()

actor = Actor(4, 2)
critic = Critic(4, 1) 
start_time = time()
actor_loss, critic_loss, rewards = train(env, actor, critic, hyperparameters=hyperparameters)
println("Total training time was $(round(time() - start_time; digits=2))")

plot_loss(actor_loss, "Actor loss")
savefig("actor_loss.png")
plot_loss(critic_loss, "Critic loss")
savefig("critic_loss.png")
plot_rewards(rewards)
