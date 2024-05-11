module DroneDiscrete

using POMDPs
using Plots
using POMDPTools

export DroneState,DroneEnv, DroneEnv, checkgoal, checkobs, checkoutbounds, plot_target, render

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

mutable struct DroneEnv <: MDP{DroneState, Symbol}
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
function POMDPs.states(env::DroneEnv)
    s = DroneState[] # initialize an array of GridWorldStates
    # loop over all our states, remeber there are two binary variables:
    # done (d)
    for d = 0:1,theta = 0.0:pi/2:3*pi/2,  y = 1:env.size[2], x = 1:env.size[1]
        push!(s, DroneState(x,y,theta,d))
    end
    return s
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

function checkterminal(env,x,y)
    x_max, y_max = env.size
    for obstacle in env.obstacles
        if x == obstacle[1] && y == obstacle[2]
            return true
        end
    end
    goal = (x == env.target[1] && y == env.target[2])
    bound = x > x_max || x < 1 || y > y_max || y < 1
    if goal || bound
        return true
    end
    return false 
end

# transition function
function POMDPs.transition(env::DroneEnv, state::DroneState, action::Symbol)
    @show state
    a = action
    x = state.x
    y = state.y
    θ = state.theta

    if state.done
        return SparseCat([DroneState(x, y, theta, true)], [1.0])
    end

    e = x+1
    b = x-1
    c = y+1
    d = y-1
    neighbors = [
        DroneState(e, y, θ, checkterminal(env,e,y))
        DroneState(b, y, θ, checkterminal(env,b,y))
        DroneState(x, c, θ, checkterminal(env,x,c))
        DroneState(x, d, θ, checkterminal(env,x,d))
        DroneState(x, y, mod(θ+pi/2,2*pi), checkterminal(env,x,y))
        DroneState(x, y, mod(maximum([θ-pi/2,θ+3*pi/2]),2*pi), checkterminal(env,x,y))
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

# is terminal
POMDPs.isterminal(env::DroneEnv, s::DroneState) = s.done


function POMDPs.stateindex(env::DroneEnv, state::DroneState)
    num_x = env.size[1]
    num_y = env.size[2]
    num_theta = 4 # There are 4 discrete values for theta

    # Define a mapping for theta values to integers
    theta_mapping = Dict(0.0 => 1, Float64(pi)/2 => 2, Float64(pi) => 3, 3*Float64(pi)/2 => 4)

    # Calculate the index based on state variables
    theta_index = theta_mapping[state.theta]
    index = state.x + (state.y - 1) * num_x + (theta_index - 1) * num_x * num_y

    # If the state is terminal, add the size of the non-terminal states
    if state.done
        index += num_x * num_y * num_theta
    end
    return index
end

function POMDPs.actionindex(env::DroneEnv, act::Symbol)
    if act==:FWD
        return 1
    elseif act==:BKWD
        return 2
    elseif act==:L
        return 3
    elseif act==:R
        return 4
    elseif act==:CCW
        return 5
    elseif act==:CW
        return 6
    end
    error("Invalid GridWorld action: $act")
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
    plot!([state.x], [state.y], mark=:diamond, markersize=20, color=:blue)
    quiver!([state.x], [state.y], quiver=[(0.5*cos(state.theta), 0.5*sin(state.theta))], color=:black, arrow=true, linewidth=5)
    
    plot!([0.5, 0.5], [0.5, 10.5], color=:black, linewidth=2)  # (0.5,0.5) to (0.5,10.5)
    plot!([0.5, 10.5], [0.5, 0.5], color=:black, linewidth=2)  # (0.5,0.5) to (10.5,0.5)
    plot!([0.5, 10.5], [10.5, 10.5], color=:black, linewidth=2)  # (0.5,10.5) to (10.5,10.5)
    plot!([10.5, 10.5], [10.5, 0.5], color=:black, linewidth=2)  # (10.5,10.5) to (10.5,0.5)

    display(p)
    println(state)
end

end