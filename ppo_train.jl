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
        "batch_size" => 10_000,
        "max_timesteps_per_episode" => 2_000,
        "updates_per_iteration" => 20,
        "total_timesteps" => 10_000_000,
        "lr" => 1e-3,
        "clip" => 0.2
    )

env = DroneEnv()

actor = Actor(4, 2)
critic = Critic(4, 1) 
actor_loss, critic_loss, rewards = train(env, actor, critic, hyperparameters=hyperparameters)

plot_loss(actor_loss, "Actor loss")
plot_loss(critic_loss, "Critic loss")
plot_rewards(rewards)
