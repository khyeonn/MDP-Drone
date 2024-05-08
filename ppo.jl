module ppo

using Flux: Chain, Dense, params, gradient, Optimise
using Distributions: Normal, logpdf
using Statistics: mean, std

export Actor, Critic

#### Actor
struct Actor
    model::Chain 
    μ::Dense
    logstd::Dense
end

## actor constructor
function Actor(input_size, output_size)
    model = Chain(
                Dense(input_size, 64, tanh),
                Dense(64, 64, tanh),
                Dense(64, output_size*2)
                )
    μ = Dense(input_size, output_size)
    logstd = Dense(input_size, output_size)

    return Actor(model, μ, logstd)
end


### get action
function get_action(actor::Actor, env)
    s = get_state(env.drone)
    x = actor.model(s)
    μ = actor.μ(s)
    σ = exp.(clamp.(actor.logstd(s), -0.5, 0.5))

    # create distribution using mean and std of action based on state
    dist = [Normal(μ[i], σ[i]) for i in 1:2]

    # sample action from distribution
    accel = clamp(rand(dist[1]), -env.max_acceleration, env.max_acceleration)
    rotate = clamp(rand(dist[2]), -env.max_rotation_rate, env.max_rotation_rate)

    a = DroneAction(accel, rotate)

    # compute log probability of sampled action
    log_prob_accel = logpdf(dist[1], accel)
    log_prob_rotate = logpdf(dist[2], rotate)

    # log_prob should have dim = num_actions
    log_prob = [log_prob_accel, log_prob_rotate]
    
    return a, log_prob
end


### rollout
function rollout(env, actor)
    max_timesteps_per_batch = 500
    max_timesteps_per_episode = 20
    t = 0

    batch_obs = Vector{DroneState}()
    batch_acts = Vector{DroneAction}()
    batch_log_probs = Vector{Vector{Float32}}()
    batch_rewards = Vector{Vector{Float32}}()
    batch_rtgo = Vector{Vector{Float32}}()
    batch_lens = Vector{Int64}()

    ep_rewards = Vector{Float32}()

    while t < max_timesteps_per_batch
        ep_rewards = Vector{Float32}()

        reset!(env)
        obs = env.drone
        done = env.isterminal
        ep_timesteps = 1

        for ep_t in 1:max_timesteps_per_episode
            ep_timesteps = ep_t
            t += 1

            push!(batch_obs, obs)
            
            action, log_prob = get_action(actor, env)
            obs, r, done = gen(env, action)

            push!(ep_rewards, r)
            push!(batch_acts, action)
            push!(batch_log_probs, log_prob)

            if done
                break
            end
        end

        push!(batch_lens, ep_timesteps)
        push!(batch_rewards, ep_rewards)
    end

    batch_rtgo = push!(batch_rtgo, compute_rtgo(batch_rewards))

    return batch_obs, batch_acts, batch_log_probs, batch_rtgo, batch_lens
end


##### compute reward to go
function compute_rtgo(batch_rewards)
    batch_rtgo = Vector{Float32}()

    for ep_rewards in Iterators.reverse(batch_rewards)
        discounted_reward = 0.0

        for reward in Iterators.reverse(ep_rewards)
            discounted_reward = reward + discounted_reward*0.99
            pushfirst!(batch_rtgo, discounted_reward)
        end

    end

    return batch_rtgo
end



##### Critic 
struct Critic
    model::Chain
end

# critic constructor
function Critic(input_size::Int, output_size::Int)
    model = Chain(
        Dense(input_size, 64, tanh),
        Dense(64, 64, tanh),
        Dense(64, output_size))

    return Critic(model)
end

function (critic::Critic)(s)
    return critic.model(s)
end

### evaluate
function evaluate(actor, batch_obs, batch_acts)
    V = reduce(vcat, [critic(get_state(obs)) for obs in batch_obs]);

    s = [get_state(obs) for obs in batch_obs];
    μ = actor.μ.(s)
    logstd = actor.logstd.(s)
    logstd = [clamp.(logstd[i], -0.5f0, 0.5f0) for i in 1:length(logstd)]
    σ = [exp.(logstd[i]) for i in 1:length(logstd)]

    # create distribution from batch observations
    dist_accel = [Normal(μ[i][1], σ[i][1]) for i in 1:length(s)]
    dist_rotate = [Normal(μ[i][2], σ[i][2]) for i in 1:length(s)]

    accel = [batch_acts[i].accel for i in 1:length(batch_acts)]
    rotate = [batch_acts[i].rotate for i in 1:length(batch_acts)]

    # compute log probs of batch actions
    log_prob_accel = [logpdf(dist_accel[i], accel[i]) for i in 1:length(batch_acts)]
    log_prob_rotate = [logpdf(dist_rotate[i], rotate[i]) for i in 1:length(batch_acts)]

    log_probs = [log_prob_accel, log_prob_rotate]

    log_probs = hcat(log_probs...)
    log_probs = reshape(log_probs, :, size(log_probs, 2))

    return V, log_probs
end


### learning update
function learn(actor, critic, env; total_timesteps = 100_000, lr = 1e-3)
    t_so_far = 0
    i_so_far = 0
    
    actor_optim = ADAM(lr)
    critic_optim = ADAM(lr)

    while t_so_far < total_timesteps
        
        batch_obs, batch_acts, batch_log_probs, batch_rtgo, batch_lens = rollout(env, actor)
        
        t_so_far += sum(batch_lens)
        i_so_far += 1
        
        V, _ = evaluate(actor, batch_obs, batch_acts)
        
        A_k = batch_rtgo[1] - deepcopy(V)
        
        
        A_k = (A_k .- mean(A_k)) ./ (std(A_k) .+ 1e-10)
        

        for _ in (1:5)
            V, curr_log_probs = evaluate(actor, batch_obs, batch_acts)

            ratios = curr_log_probs - transpose(hcat(batch_log_probs...))

            # surrogate objectives
            surr1 = ratios .* A_k
            surr2 = clamp.(ratios, 0.8f0, 1.2f0) .* A_k

            actor_loss = -mean(min.(surr1, surr2))
            critic_loss = Flux.mse(V, batch_rtgo[1])

            Flux.reset!(actor_optim)
            gs = gradient(() -> actor_loss, Flux.params(actor.model))
            Flux.update!(actor_optim, Flux.params(actor.model), gs)

            
            Flux.reset!(critic_optim)
            gs = gradient(() -> critic_loss, Flux.params(critic.model))
            Flux.update!(critic_optim, Flux.params(critic.model), gs)

        end

    end

end


end # module