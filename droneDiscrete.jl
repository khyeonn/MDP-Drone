module Drone

using POMDPs
using Plots

export DroneEnv, DroneState, DroneAction

# default values for DroneEnv
const DEFAULT_SIZE = (50.0, 50.0)
const STEP_SIZE = 0.001
const DEFAULT_DISCOUNT = 0.95

struct DroneState
    x::Int64
    y::Int64
    theta::Float64 # heading angle
end

struct DroneAction #From the drone frame
    :FWD::Symbol
    :BKWD::Symbol
    :L::Symbol
    :R::Symbol
    :CCW::Symbol
    :CW::Symbol
end

struct DroneEnv
    size::Tuple{Int, Int} 
    target::Tuple{Int64, Int64} # (x, y)
    discount::Float64
    isterminal::Bool
    # obstacles::Vector{Tuple{Float64, Float64, Float64}} # for later
end

# environment constructor
function DroneEnv(;
        size = DEFAULT_SIZE,
        discount = DEFAULT_DISCOUNT
        )

    target_x, target_y = rand(1:STEP_SIZE:size[1]), rand(1:STEP_SIZE:size[2])
    target = (target_x, target_y)
    isterminal = false

    return DroneEnv(size, target, discount, isterminal)
end


# transition function
function POMDPs.transition(env::DroneEnv, state::DroneState, action::DroneAction)
    # update heading angle
    theta_new = state.theta + clamp(action.rotate, -env.max_rotation_rate, env.max_rotation_rate)
    theta_new = atan(sin(theta_new), cos(theta_new)) # limit to -π to π 

    # velocity update
    v_new = clamp(state.v + action.accel, -env.max_velocity, env.max_velocity)

    # update position
    x_new = state.x + v_new*cos(theta_new)
    y_new = state.y + v_new*sin(theta_new)

    new_state = DroneState(x_new, y_new, v_new, theta_new)

    if isterminal(env, new_state)
        env.isterminal = true
    end

    return new_state
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