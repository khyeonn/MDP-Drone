module ppo

include("Drone.jl")
using ..Drone: DroneEnv, DroneAction, DroneState, get_state, isterminal, discount, gen, reset!, create_animation

using Plots
using Flux: Chain, Dense, params, gradient, Optimise, Adam, mse, update!, leakyrelu, ignore
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
    actor_loss
    critic_loss
    hyperparameters::Dict{String, Real}
end

# init default hyperparameters
function PPO(env::DroneEnv, actor::Actor, critic::Critic;
    hyperparameters = Dict(
        "batch_size" => 200,
        "max_timesteps_per_episode" => 100,
        "updates_per_iteration" => 5,
        "total_timesteps" => 100_000,
        "lr" => 1e-3,
        "clip" => 0.2
    ))
    actor_loss = Vector{Float32}()
    critic_loss = Vector{Float32}()

    return PPO(env, actor, critic, actor_loss, critic_loss, hyperparameters)
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
    # apply tanh to clamp between -1 and 1
    accel = tanh(rand(dist[1]))
    rotate = tanh(rand(dist[2]))

    # compute log probability of sampled action
    log_prob_accel = logpdf(dist[1], accel)
    log_prob_rotate = logpdf(dist[2], rotate)

    # log_prob should have dim = num_actions
    log_prob = [log_prob_accel, log_prob_rotate]
    
    a = DroneAction(accel, rotate)
    return a, log_prob
end


### rollout
function rollout(ppo::PPO)
    batch_size = ppo.hyperparameters["batch_size"]
    max_timesteps_per_episode = ppo.hyperparameters["max_timesteps_per_episode"]

    t = 0

    batch_obs = Vector{DroneState}()
    batch_acts = Vector{DroneAction}()
    batch_log_probs = Vector{Vector{Float32}}()
    batch_rewards = Vector{Vector{Float32}}()
    batch_rtgo = Vector{Float32}()
    batch_lens = Vector{Int64}()
    batch_dones = 0

    batch_env = Vector{DroneEnv}()

    ep_rewards = Vector{Float32}()

    while length(batch_obs) <= batch_size
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
                    batch_dones += 1
                    # animation = create_animation(batch_env)
                    # gif(animation, "animations/rollout.gif")
                end
                break
            end

            if length(batch_obs) >= batch_size
                break
            end

        end
        push!(batch_lens, ep_timesteps)
        push!(batch_rewards, ep_rewards)
    end
    batch_rtgo = compute_rtgo(ppo.env, batch_rewards)

    return batch_obs, batch_acts, batch_log_probs, batch_rtgo, batch_lens, batch_dones
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

    function actor_loss_fn(ppo::PPO, batch_obs, batch_acts, batch_log_probs, batch_rtgo)
        clip = ppo.hyperparameters["clip"]

        # get model parameters
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

        # compute value and advantage
        V = reduce(vcat, [ppo_network.critic(get_state(obs)) for obs in batch_obs])
        V_scaled = V*maximum(abs.(batch_rtgo)) / maximum(abs.(V))
        
        A_k = batch_rtgo - deepcopy(V_scaled)
        A_k = (A_k .- mean(A_k)) ./ (std(A_k) .+ 1e-10)

        ratios = exp.(log_probs - transpose(hcat(batch_log_probs...)))

        # compute surrogate objectives
        surr1 = ratios .* A_k
        surr2 = clamp.(ratios, 1-clip, 1+clip) .* A_k

        # compute actor loss
        actor_loss = -mean(min.(surr1, surr2))

        return actor_loss
    end

    function critic_loss_fn(ppo::PPO, batch_obs, batch_rtgo)
        # compute value estimate
        V = reduce(vcat, [ppo.critic(get_state(obs)) for obs in batch_obs])
        V_scaled = V*maximum(abs.(batch_rtgo)) / maximum(abs.(V))

        # compute critic loss
        critic_loss = mse(V_scaled, batch_rtgo)

        return critic_loss
    end


    total_timesteps = ppo_network.hyperparameters["total_timesteps"]
    updates_per_iteration = ppo_network.hyperparameters["updates_per_iteration"]
    lr = ppo_network.hyperparameters["lr"]
    clip = ppo_network.hyperparameters["clip"]

    t_so_far = 0
    i_so_far = 0
    dones = 0

    policy_rewards = Vector{Vector{Float32}}()
    
    while t_so_far < total_timesteps # train for T timesteps (can also do epochs)
        start_time = time()

        # perform rollout to collect experience batch
        batch_obs, batch_acts, batch_log_probs, batch_rtgo, batch_lens, batch_dones = rollout(ppo_network)

        dones += batch_dones
        t_so_far += sum(batch_lens)
        i_so_far += 1
    
        # update actor and critic network parameters
        for _ in 1:updates_per_iteration
            actor_opt = Adam(lr)
            actor_gs = gradient(() -> actor_loss_fn(ppo_network, batch_obs, batch_acts, batch_log_probs, batch_rtgo), params(ppo_network.actor.model, ppo_network.actor.μ, ppo_network.actor.logstd))
            update!(actor_opt, params(ppo_network.actor.model, ppo_network.actor.μ, ppo_network.actor.logstd), actor_gs)

            critic_opt = Adam(lr)
            critic_gs = gradient(() -> critic_loss_fn(ppo_network, batch_obs, batch_rtgo), params(ppo_network.critic.model))
            update!(critic_opt, params(ppo_network.critic.model), critic_gs)

            ignore() do
                push!(ppo_network.actor_loss, actor_loss_fn(ppo_network, batch_obs, batch_acts, batch_log_probs, batch_rtgo))
                push!(ppo_network.critic_loss, critic_loss_fn(ppo_network, batch_obs, batch_rtgo))
            end
        end


        if i_so_far % 500 == 0
            batch_env, rewards = _run(ppo_network)
            push!(policy_rewards, rewards)

            animation = create_animation(batch_env)
            gif(animation, "animations/ppo_$i_so_far.gif")
            @save "models/ppo_$i_so_far.bson" ppo_network
        end

        elapsed_time = time() - start_time
        avg_ep_lens = mean(batch_lens)
        avg_ep_reward = mean(batch_rtgo)
        avg_actor_loss = mean(ppo_network.actor_loss)
        avg_critic_loss = mean(ppo_network.critic_loss)

        ##### logging
        println("----------- Iteration #$i_so_far -----------")
        println("Iteration took $(round(elapsed_time; digits=2)) seconds")
        println("Average episodic length: $(round(avg_ep_lens; digits=2))")
        println("Average episodic reward: $(round(avg_ep_reward; digits=3))")
        println("Timesteps so far: $t_so_far")
        println("Solved in rollout $dones times")
        println("Average actor loss : $(round(avg_actor_loss; digits=3))")
        println("Average critic loss: $(round(avg_critic_loss; digits=3))")
        println("----------- END SUMMARY -----------\n")

        @save "models/ppo_final.bson" ppo_network
    end
    return ppo_network.actor_loss, ppo_network.critic_loss, policy_rewards
end


end # module