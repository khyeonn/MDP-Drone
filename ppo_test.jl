if !@isdefined Drone
    include("Drone.jl")
    include("ppo.jl")
end
using .Drone: DroneEnv, DroneAction, plot_frame, reset!, gen, create_animation, isterminal
using .ppo: PPO, Actor, Critic, get_action
using Statistics: mean
using Plots
using BSON: @load

@load "models/ppo_final.bson" ppo_network


function rollout(ppo::PPO; max_episodes = 1000)
    rewards = Vector{Float32}()
    batch_lens = Vector{Int64}()
    episode_env = Vector{DroneEnv}()
    batch_env = Vector{Vector{DroneEnv}}()
    episode_acts = Vector{DroneAction}()
    batch_acts = Vector{Vector{DroneAction}}()

    for _ in 1:max_episodes
        reset!(ppo.env)
        empty!(episode_env)
        empty!(episode_acts)

        ep_rewards = 0.0
        ep_len = 0

        for t in 1:ppo.hyperparameters["max_timesteps_per_episode"]
            action, _ = get_action(ppo)
            _, r, done = gen(ppo.env, action)

            push!(episode_env, deepcopy(ppo.env))
            push!(episode_acts, deepcopy(action))

            ep_rewards += r
            ep_len = t

            if done
                break
            end
        end
        push!(batch_env, deepcopy(episode_env))
        push!(batch_acts, episode_acts)
        push!(rewards, ep_rewards)
        push!(batch_lens, ep_len)
    end
    return rewards, batch_lens, batch_env, batch_acts
end

rewards, ep_lengths, batch_env, batch_acts = rollout(ppo_network)
p = plot(1:length(rewards), rewards, xlabel="Episodes", ylabel="Total rewards")
display(p)

i = argmax(rewards)
animation = create_animation(batch_env[i])
gif(animation, "gif_test.gif")