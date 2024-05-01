using POMDPs

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

struct DroneMDP
    max_velocity::Float64
    max_rotation_speed::Float64
    discount::Float64
end


function POMDPs.transition(mdp::DroneMDP, state::DroneState, action::DroneAction)
    # velocity update
    v_new = clamp(state.v + a.accel, -mdp.max_velocity, mdp.max_velocity)

    # update position
    x_new = state.x + v_new*cos(state.theta)
    x_new = state.y + v_new*sin(state.theta)

    # update heading angle
    theta_new = state.theta + a.rotate
    theta_new = atan(sin(theta_new), cos(theta_new)) # limit to -π to π 

    return DroneState(x_new, y_new, v_new, theta_new)
end

function POMDPs.reward(mdp::DroneMD, state::DroneState, action::DroneAction)
    
end