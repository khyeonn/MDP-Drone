module Drone

using POMDPs
using Plots

export DroneEnv, DroneState, DroneAction

# default values for DroneEnv
const DEFAULT_SIZE = (50.0, 50.0)
const STEP_SIZE = 0.001
const DEFAULT_DISCOUNT = 0.95
const DEFAULT_PROB = 0.7

struct DroneState
    x::Int64
    y::Int64
    theta::Float64 # heading angle
    done::Bool # are we in a terminal state?
end

mutable struct DroneEnv
    size::Tuple{Int, Int} 
    target::Tuple{Int64, Int64} # (x, y)
    discount::Float64
    tprob::Float64 # probability of transitioning to the desired state
    # obstacles::Vector{Tuple{Float64, Float64, Float64}} # for later
end

# environment constructor
function DroneEnv(;
        size = DEFAULT_SIZE,
        discount = DEFAULT_DISCOUNT,
        tprob = DEFAULT_PROB
        )

    target_x, target_y = rand(1:STEP_SIZE:size[1]), rand(1:STEP_SIZE:size[2])
    target = (target_x, target_y)
    isterminal = false

    return DroneEnv(size, target, discount, isterminal)
end

# action space
POMDPs.actions(env::DroneEnv) = [:FWD, :BKWD, :L, :R, :CCW, :CW]

# Check if its on the goal state
checkgoal(env::DroneEnv, s::DroneState) = s.x == env.target[1] && s.y == env.target[2]


# transition function
function POMDPs.transition(env::DroneEnv, state::DroneState, action::Symbol)
    a = action
    x = state.x
    y = state.y
    θ = state.theta

    if state.done
        return SparseCat([DroneState(x, y, theta, true)], [1.0])
    end

    ingoal = checkgoal(env, state)

    neighbors = [
        DroneState(x+1, y, θ, ingoal) 
        DroneState(x-1, y, θ, ingoal)
        DroneState(x, y+1, θ, ingoal)
        DroneState(x, y-1, θ, ingoal)
        DroneState(x, y, mod(θ+pi/2,2*pi/2), ingoal) # set limit
        DroneState(x, y, mod(maximum([θ-pi/2,θ+3*pi/2]),2*pi), ingoal) # set limit
        ]

    probability = fill((1-env.tprob)/5, 6)
    if a == :CW
        probability[6] = env.tprob
    elseif a == :CCW
        probability[5] = env.tprob
    end

    if a == :FWD
        if state.theta == 0.0
            probability[1] = env.tprob
        elseif state.theta == pi/2
            probability[3] = env.tprob
        elseif state.theta == pi
            probability[2] = env.tprob
        elseif state.theta == 3*pi/2
            probability[4] = env.tprob
        end
    elseif a == :BKWD
        if state.theta == 0.0
            probability[2] = env.tprob
        elseif state.theta == pi/2
            probability[4] = env.tprob
        elseif state.theta == pi
            probability[1] = env.tprob
        elseif state.theta == 3*pi/2
            probability[3] = env.tprob
        end
    elseif a == :L
        if state.theta == 0.0
            probability[3] = env.tprob
        elseif state.theta == pi/2
            probability[2] = env.tprob
        elseif state.theta == pi
            probability[4] = env.tprob
        elseif state.theta == 3*pi/2
            probability[1] = env.tprob
        end
    elseif a == :R
        if state.theta == 0.0
            probability[4] = env.tprob
        elseif state.theta == pi/2
            probability[1] = env.tprob
        elseif state.theta == pi
            probability[3] = env.tprob
        elseif state.theta == 3*pi/2
            probability[2] = env.tprob
        end

    return SparseCat(neighbors, probability)
end

# terminal condition
function isterminal(env::DroneEnv, state::DroneState)
    distance_to_target = sqrt((state.x - env.target[1])^2 + (state.y - env.target[2])^2)

    return distance_to_target <= env.target[3]
end

# reward function
function POMDPs.reward(env::DroneEnv, state::DroneState)
    if isterminal(env, state)
        return 100.0
    else
        return -0.1
    end
end

# discount
POMDPs.discount(env::DroneEnv) = env.discount


# generate next state sp, and reward r 
function gen(env::DroneEnv, state::DroneState, action::DroneAction)
    sp = POMDPs.transition(env, state, action)
    r = POMDPs.reward(env, state)

    return sp, r
end

function plot_target(p::Plots.Plot, x, y, r; color::Symbol, label::String)
    n = 100
    rads = range(0, stop=2*π, length=n)
    x = x .+ r.*cos.(rads)
    y = y .+ r.*sin.(rads)
    plot!(p, x, y, label=label, color=color, legend=true)
end

# render
function render(env::DroneEnv, state::DroneState)
    p = plot(size=(800, 800), xlim=(0, env.size[1]), ylim=(0, env.size[2]), legend=false)

    plot_target(p, env.target[1], env.target[2], env.target[3], label="Target Region", color=:red)

    plot!([state.x], [state.y], mark=:circle, markersize=5, color=:blue, label="Drone")

    display(p)
end


end # module