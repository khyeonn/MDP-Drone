if !@isdefined Drone
    include("Drone.jl")
    include("ppo.jl")
end
using .Drone: DroneEnv, plot_frame, reset!, gen, create_animation
using .ppo: PPO, Actor, Critic, get_action
using Statistics: mean
using Plots
using BSON: @load

@load "models/ppo_final.bson" ppo_network


function rollout(ppo::PPO; max_episodes = 100)
    rewards = Vector{Float32}()
    total_len = Vector{Int64}()
    episode_env = Vector{DroneEnv}()
    batch_env = Vector{Vector{DroneEnv}}()

    for _ in 1:max_episodes
        reset!(ppo.env)
        empty!(episode_env)

        ep_rewards = 0.0
        ep_len = 0

        for t in 1:ppo.hyperparameters["max_timesteps_per_episode"]
            action, _ = get_action(ppo)
            _, r, done = gen(ppo.env, action)

            push!(episode_env, deepcopy(ppo.env))

            ep_rewards += r
            ep_len = t

            if done
                break
            end
        end
        push!(batch_env, episode_env)
        push!(rewards, ep_rewards)
        push!(total_len, ep_len)
    end
    return rewards, total_len, batch_env
end

rewards, ep_lengths, batch_env = rollout(ppo_network)

i = argmax(rewards)
animation = create_animation(batch_env[i])
gif(animation, "gif_test.gif")