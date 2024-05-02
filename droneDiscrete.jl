module DroneDiscrete

using POMDPs
using Plots
using POMDPTools

export DroneState,DroneEnv, DroneEnv, checkgoal, checkobs, checkoutbounds, plot_target, render, gen

# default values for DroneEnv
const DEFAULT_SIZE = (10, 10)
const DEFAULT_DISCOUNT = 0.95
const DEFAULT_PROB = 0.7

struct DroneState
    x::Int64
    y::Int64
    theta::Float64 # heading angle
    done::Bool # are we in a terminal state?
end

DroneState(x::Int64, y::Int64, θ::Float64) = DroneState(x,y,θ,false)

mutable struct DroneEnv
    size::Tuple{Int64, Int64} 
    obstacles::Vector{Tuple{Int64, Int64}}
    target::Tuple{Int64, Int64} # (x, y)
    discount::Float64
    tprob::Float64 # probability of transitioning to the desired state
    
end

# environment constructor
function DroneEnv(;
        target = (10,10),
        size = DEFAULT_SIZE,
        discount = DEFAULT_DISCOUNT,
        tprob = DEFAULT_PROB
        )

    obstacles = [(5, 10), (5, 9), (10, 9),(2,8),(3,8),(10,8),(2,7),(3,7),(10,7),(2,6),(3,6),(6,6),(7,6),(8,6),(9,6),(10,6),(10,5),(10,4),(2,3),(3,3),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2)]
    return DroneEnv(size,obstacles, target, discount, tprob,)
end

# action space
POMDPs.actions(env::DroneEnv) = [:FWD, :BKWD, :L, :R, :CCW, :CW]

# Check if current state is the goal state
checkgoal(env::DroneEnv, s::DroneState) = s.x == env.target[1] && s.y == env.target[2]
# Check if current state is obstacle
function checkobs(env::DroneEnv, s::DroneState)
    obstacles = env.obstacles
    for obstacle in obstacles
        if s.x == obstacle[1] && s.y == obstacle[2]
            return true
        end
    end
    return false
end
# Check if current state is out of bounds
function checkoutbounds(env::DroneEnv, s::DroneState)
    x_max, y_max = env.size
    if s.x > x_max || s.x < 1 || s.y > y_max || s.y < 1
        return true
    end
    return false
end

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
        DroneState(x, y, mod(θ+pi/2,2*pi), ingoal) # set limit
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
    end

    return SparseCat(neighbors, probability)
end

# reward function
function POMDPs.reward(env::DroneEnv, state::DroneState, action::Symbol)
    if checkgoal(env, state)
        return 100.0
    elseif checkobs(env, state)
        return -100.0
    elseif checkoutbounds(env,state)
        return -100.0
    else
        return -0.1
    end
end

# discount
POMDPs.discount(env::DroneEnv) = env.discount


# generate next state sp, and reward r 
function gen(env::DroneEnv, state::DroneState, action::Symbol)
    sp = POMDPs.transition(env, state, action)
    r = POMDPs.reward(env, state, action)

    return sp, r
end


# render
function render(env::DroneEnv, state::DroneState)
    p = plot(size=(800, 800), xlim=(0, env.size[1]+1), ylim=(0, env.size[2]+1), legend=false)
    xticks!(0:0.5:env.size[1]+1)
    yticks!(0:0.5:env.size[2]+1)
    for obs in env.obstacles
        plot!([obs[1]], [obs[2]], mark=:circle, markersize=20, color=:black)
    end
    plot!([env.target[1]], [env.target[2]], mark=:star, markersize=20, color=:yellow)
    plot!([state.x], [state.y], mark=:circle, markersize=20, color=:blue)
    
    plot!([0.5, 0.5], [0.5, 10.5], color=:black, linewidth=2)  # (0.5,0.5) to (0.5,10.5)
    plot!([0.5, 10.5], [0.5, 0.5], color=:black, linewidth=2)  # (0.5,0.5) to (10.5,0.5)
    plot!([0.5, 10.5], [10.5, 10.5], color=:black, linewidth=2)  # (0.5,10.5) to (10.5,10.5)
    plot!([10.5, 10.5], [10.5, 0.5], color=:black, linewidth=2)  # (10.5,10.5) to (10.5,0.5)

    display(p)
end

end