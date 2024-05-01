module Drone

using POMDPs

export DroneEnv, DroneState, DroneAction

const DEFAULT_SIZE = (50.0, 50.0)

struct DroneState
    x::Float64 
    y::Float64
    v::Float64
    theta::Float64 # heading angle
end

struct DroneAction
    accel::Float64
    rotate::Float64 # radians
end

struct DroneEnv
    size::Tuple{Int, Int}
    goal::Tuple{Float64, Float64, Float64}
    max_velocity::Float64
    max_rotation_speed::Float64
    discount::Float64
    # obstacles::Vector{Tuple{Float64, Float64, Float64}}
end

function DroneEnv(;size = DEFAULT_SIZE,
        max_velocity = 1.0,
        max_rotation_speed = 0.1,
        discount = 0.95,
        goal_radius = 5.0
        )
    step_size = 0.001
    goal_x, goal_y = rand(1:step_size:size[1]), rand(1:step_size:size[2])
    goal = (goal_x, goal_y, goal_radius)
    return DroneEnv(size, goal, max_velocity, max_rotation_speed, discount)
end


function POMDPs.transition(env::DroneEnv, state::DroneState, action::DroneAction)
    # update heading angle
    theta_new = state.theta + clamp(action.rotate, -env.max_rotation_speed, env.max_rotation_speed)
    theta_new = atan(sin(theta_new), cos(theta_new)) # limit to -π to π 

    # velocity update
    v_new = clamp(state.v + action.accel, -env.max_velocity, env.max_velocity)

    # update position
    x_new = state.x + v_new*cos(theta_new)
    y_new = state.y + v_new*sin(theta_new)

    return DroneState(x_new, y_new, v_new, theta_new)
end

function POMDPs.reward(state::DroneState, target)
    distance_to_target = sqrt((state.x - target[1])^2 + (state.y - target[2])^2)

    if distance_to_target < target[3]
        return 100.0
    else
        return -0.1
    end
end

end # module