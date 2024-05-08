if !@isdefined Drone
    include("Drone.jl")
    include("ppo.jl")
end
using POMDPs
using Plots
using .Drone: DroneEnv, render
using .ppo: PPO, Actor, Critic, learn

function train(env, actor, critic; hyperparameters)
    model = PPO(env, actor, critic)

    if !isempty(hyperparameters)
        ppo._init_hyperparameters(model, hyperparameters)
    end

    actor_loss, critic_loss = learn(model)

    return actor_loss, critic_loss
end

function plot_loss(data, label::String)
    p = plot(1:length(data), data, label=label)
    display(p)
end

hyperparameters = Dict(
        "max_timesteps_per_batch" => 2000,
        "max_timesteps_per_episode" => 500,
        "total_timesteps" => 200_000,
        "lr" => 1e-5,
        "clip" => 0.2
    )

env = DroneEnv()

actor = Actor(4, 2)
critic = Critic(4, 1)
actor_loss, critic_loss = train(env, actor, critic, hyperparameters=hyperparameters)

plot_loss(actor_loss, "Actor loss")
plot_loss(critic_loss, "Critic loss")
