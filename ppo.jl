module ppo

include("Drone.jl")
using ..Drone: DroneEnv, DroneAction, DroneState, get_state, isterminal, discount, gen, reset!, create_animation

using Plots
using Flux: Chain, Dense, params, gradient, Optimise, Adam, mse, update!, leakyrelu
using Distributions: Normal, logpdf
using Statistics: mean, std
using BSON: @save

export PPO, Actor, Critic

#### Actor
mutable struct Actor
    model::Chain 
    μ::Dense
    logstd::Dense
end

## actor constructor
function Actor(input_size::Int, output_size::Int)
    model = Chain(
                Dense(input_size, 64, tanh),
                Dense(64, 64, tanh),
                Dense(64, output_size*2)
                )
    μ = Dense(input_size, output_size)
    logstd = Dense(input_size, output_size)

    return Actor(model, μ, logstd)
end

##### Critic 
mutable struct Critic
    model::Chain
end

# critic constructor
function Critic(input_size::Int, output_size::Int)
    model = Chain(
        Dense(input_size, 64, leakyrelu),
        Dense(64, 64, leakyrelu),
        Dense(64, output_size))

    return Critic(model)
end

# used to query critic for value
function (critic::Critic)(s)
    return critic.model(s)
end

### PPO
mutable struct PPO
    env::DroneEnv
    actor::Actor
    critic::Critic
    hyperparameters::Dict{String, Real}
end

# init default hyperparameters
function PPO(env::DroneEnv, actor::Actor, critic::Critic;
    hyperparameters = Dict(
        "max_timesteps_per_batch" => 1000,
        "max_timesteps_per_episode" => 100,
        "updates_per_iteration" => 5,
        "total_timesteps" => 100_000,
        "lr" => 1e-3,
        "clip" => 0.2
    ))

    return PPO(env, actor, critic, hyperparameters)
end

function _init_hyperparameters(ppo::PPO, hyperparameters::Dict{String, Real})
    merge!(ppo.hyperparameters, hyperparameters)
end


### get action
function get_action(ppo::PPO)
    s = get_state(ppo.env.drone)
    x = ppo.actor.model(s)
    μ = ppo.actor.μ(x)
    σ = exp.(clamp.(ppo.actor.logstd(x), -5.0, 5.0))

    # create distribution using mean and std of action based on state
    dist = [Normal(μ[i], σ[i]) for i in 1:2]

    # sample action from distribution
    accel = tanh(rand(dist[1]))
    rotate = tanh(rand(dist[2]))

    a = DroneAction(accel, rotate)

    # compute log probability of sampled action
    log_prob_accel = logpdf(dist[1], accel)
    log_prob_rotate = logpdf(dist[2], rotate)

    # log_prob should have dim = num_actions
    log_prob = [log_prob_accel, log_prob_rotate]
    
    return a, log_prob
end


### rollout
function rollout(ppo::PPO)
    max_timesteps_per_batch = ppo.hyperparameters["max_timesteps_per_batch"]
    max_timesteps_per_episode = ppo.hyperparameters["max_timesteps_per_episode"]

    t = 0

    batch_obs = Vector{DroneState}()
    batch_acts = Vector{DroneAction}()
    batch_log_probs = Vector{Vector{Float32}}()
    batch_rewards = Vector{Vector{Float32}}()
    batch_rtgo = Vector{Float32}()
    batch_lens = Vector{Int64}()

    batch_env = Vector{DroneEnv}()

    ep_rewards = Vector{Float32}()

    while t < max_timesteps_per_batch
        ep_rewards = Vector{Float32}()

        reset!(ppo.env)
        empty!(batch_env)
        push!(batch_env, deepcopy(ppo.env))
        obs = ppo.env.drone
        done = ppo.env.isterminal
        ep_timesteps = 1

        for ep_t in 1:max_timesteps_per_episode
            ep_timesteps = ep_t
            t += 1

            push!(batch_obs, obs)
            
            action, log_prob = get_action(ppo)
            obs, r, done = gen(ppo.env, action)

            push!(batch_env, deepcopy(ppo.env))
            push!(ep_rewards, r)
            push!(batch_acts, action)
            push!(batch_log_probs, log_prob)

            if done
                if isterminal(ppo.env) == 1
                    println("Solved in rollout")
                    # animation = create_animation(batch_env)
                    # gif(animation, "animations/rollout.gif")
                end
                break
            end
        end
        push!(batch_lens, ep_timesteps)
        push!(batch_rewards, ep_rewards)
    end
    batch_rtgo = compute_rtgo(ppo.env, batch_rewards)

    return batch_obs, batch_acts, batch_log_probs, batch_rtgo, batch_lens
end


##### compute reward to go
function compute_rtgo(env, batch_rewards)
    batch_rtgo = Vector{Float32}()

    for ep_rewards in Iterators.reverse(batch_rewards)
        discounted_reward = 0.0

        for reward in Iterators.reverse(ep_rewards)
            discounted_reward = reward + discounted_reward*env.discount
            pushfirst!(batch_rtgo, discounted_reward)
        end
    end

    return batch_rtgo
end


### evaluate
function evaluate(ppo::PPO, batch_obs, batch_acts)
    V = reduce(vcat, [ppo.critic(get_state(obs)) for obs in batch_obs])

    s = [get_state(obs) for obs in batch_obs];
    x = ppo.actor.model.(s)
    μ = ppo.actor.μ.(x)
    logstd = ppo.actor.logstd.(x)
    logstd = [clamp.(logstd[i], -5.0f0, 5.0f0) for i in eachindex(logstd)]
    σ = [exp.(logstd[i]) for i in eachindex(logstd)]

    # create distribution from batch observations
    dist_accel = [Normal(μ[i][1], σ[i][1]) for i in eachindex(x)]
    dist_rotate = [Normal(μ[i][2], σ[i][2]) for i in eachindex(x)]

    accel = [batch_acts[i].accel for i in eachindex(batch_acts)]
    rotate = [batch_acts[i].rotate for i in eachindex(batch_acts)]

    # compute log probs of batch actions
    log_prob_accel = [logpdf(dist_accel[i], accel[i]) for i in eachindex(batch_acts)]
    log_prob_rotate = [logpdf(dist_rotate[i], rotate[i]) for i in eachindex(batch_acts)]

    log_probs = [log_prob_accel, log_prob_rotate]

    # reshape 
    log_probs = hcat(log_probs...)
    log_probs = reshape(log_probs, :, size(log_probs, 2))

    return V, log_probs
end


### run model
function _run(ppo::PPO)
    max_timesteps_per_episode = ppo.hyperparameters["max_timesteps_per_episode"]
    batch_env = Vector{DroneEnv}()
    ep_rewards = Vector{Float32}()

    reset!(ppo.env)
    push!(batch_env, deepcopy(ppo.env))

    for _ in 1:max_timesteps_per_episode
        action, _ = get_action(ppo)
        _, r, done = gen(ppo.env, action)

        push!(batch_env, deepcopy(ppo.env))
        push!(ep_rewards, r)

        if done
            break
        end
    end

    return batch_env, ep_rewards
end


### learning update
function learn(ppo_network::PPO)
    total_timesteps = ppo_network.hyperparameters["total_timesteps"]
    updates_per_iteration = ppo_network.hyperparameters["updates_per_iteration"]
    lr = ppo_network.hyperparameters["lr"]
    clip = ppo_network.hyperparameters["clip"]

    t_so_far = 0
    i_so_far = 0

    actor_losses = Vector{Float32}()
    critic_losses = Vector{Float32}()
    policy_rewards = Vector{Vector{Float32}}()
    
    while t_so_far < total_timesteps
        start_time = time()

        batch_obs, batch_acts, batch_log_probs, batch_rtgo, batch_lens = rollout(ppo_network)
        
        t_so_far += sum(batch_lens)
        i_so_far += 1
        
        V, _ = evaluate(ppo_network, batch_obs, batch_acts)
        
        A_k = batch_rtgo - deepcopy(V)
        A_k = (A_k .- mean(A_k)) ./ (std(A_k) .+ 1e-10)
    
        for _ in (1:updates_per_iteration)
            V, curr_log_probs = evaluate(ppo_network, batch_obs, batch_acts)

            ratios = exp.(curr_log_probs - transpose(hcat(batch_log_probs...)))

            # surrogate objectives
            surr1 = ratios .* A_k
            surr2 = clamp.(ratios, 1-clip, 1+clip) .* A_k

            actor_loss = -mean(min.(surr1, surr2))
            critic_loss = mse(V, batch_rtgo)

            actor_opt = Adam(lr)
            actor_gs = gradient(() -> actor_loss, params(ppo_network.actor.model, ppo_network.actor.μ, ppo_network.actor.logstd))
            update!(actor_opt, params(ppo_network.actor.model, ppo_network.actor.μ, ppo_network.actor.logstd), actor_gs)

            critic_opt = Adam(lr)
            critic_gs = gradient(() -> critic_loss, params(ppo_network.critic.model))
            update!(critic_opt, params(ppo_network.critic.model), critic_gs)

            push!(actor_losses, actor_loss)
            push!(critic_losses, critic_loss)
        end

        if i_so_far % 50 == 0
            batch_env, rewards = _run(ppo_network)
            push!(policy_rewards, rewards)

            animation = create_animation(batch_env)
            gif(animation, "animations/ppo_$i_so_far.gif")
            @save "models/ppo_$i_so_far.bson" ppo_network
        end

        elapsed_time = time() - start_time
        avg_ep_lens = mean(batch_lens)
        avg_ep_reward = mean(batch_rtgo[1])

        ##### logging
        println("----------- Iteration #$i_so_far -----------")
        println("Iteration took $(round(elapsed_time; digits=2)) seconds")
        println("Average episodic length: $(round(avg_ep_lens; digits=2))")
        println("Average episodic reward: $(round(avg_ep_reward; digits=3))")
        println("Timesteps so far: $t_so_far")
        println("----------- END SUMMARY -----------\n")

        @save "models/ppo_final.bson" ppo_network
    end
    return actor_losses, critic_losses, policy_rewards
end


end # module