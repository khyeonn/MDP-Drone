module Drone

using POMDPs
using Plots
using Random
using LinearAlgebra

export DroneEnv, DroneState, DroneAction, Target, Obstacle

# default values for DroneEnv
const DEFAULT_SIZE = (50.0, 50.0)
const DEFAULT_TARGET_RADIUS = 5.0
const STEP_SIZE = 0.001
const DEFAULT_DISCOUNT = 0.99
const DEFAULT_MAX_VELOCITY = 1.0 
const DEFAULT_MAX_ACCELERATION = 1.0
const DEFAULT_MAX_ROTATION_RATE = 0.05 
const DEFAULT_OBSTACLES = 10
const DEFAULT_STATE = (2.0, 2.0, 0.0, 0.0)

struct DroneState
    x::Float32 
    y::Float32
    v::Float32
    theta::Float32 # heading angle
end

struct DroneAction
    accel::Float32
    rotate::Float32 # radians
end

struct Target
    x::Float64
    y::Float64
    r::Float64
end

struct Obstacle
    x::Float64
    y::Float64
    r::Float64
end

mutable struct DroneEnv
    size::Tuple{Float64, Float64} 
    drone::DroneState
    target::Target 
    max_velocity::Float32
    max_acceleration::Float32
    max_rotation_rate::Float32
    discount::Float32
    isterminal::Bool
    num_obstacles::Int64
    obstacles::Vector{Obstacle}
    initial_state::DroneState
end

# environment constructor
function DroneEnv(;
        size = DEFAULT_SIZE,
        drone = DroneState(DEFAULT_STATE...),
        target_radius = DEFAULT_TARGET_RADIUS,
        max_velocity = DEFAULT_MAX_VELOCITY,
        max_acceleration = DEFAULT_MAX_ACCELERATION,
        max_rotation_rate = DEFAULT_MAX_ROTATION_RATE,
        discount = DEFAULT_DISCOUNT,
        num_obstacles = DEFAULT_OBSTACLES
        )
    initial_state = drone
    ## for testing purposes
    Random.seed!(1)

    ## set random target location
    target_x, target_y = rand(1:STEP_SIZE:size[1]), rand(1:STEP_SIZE:size[2])
    target = Target(target_x, target_y, target_radius)

    ## define random obstacles
    obstacles = generate_obstacles(size, num_obstacles, target)

    isterminal = false

    return DroneEnv(size, drone, target, max_velocity, max_acceleration, max_rotation_rate, discount, isterminal, num_obstacles, obstacles, initial_state)
end


## generate random obstacles
function generate_obstacles(size::Tuple{Float64, Float64}, num_obstacles::Int64, target::Target)
    obstacles = Vector{Obstacle}()
    min_radius= 2.5
    max_radius = 7.5

    for _ in 1:num_obstacles
        obstacle_x, obstacle_y = rand(1:STEP_SIZE:size[1]), rand(1:STEP_SIZE:size[2])
        obstacle_r = rand(min_radius:STEP_SIZE:max_radius)

        ## if target and obstacle regions overlap, generate new obstacle
        while norm([target.x, target.y] - [obstacle_x, obstacle_y]) < target.r + obstacle_r
            obstacle_x, obstacle_y = rand(1:STEP_SIZE:size[1]), rand(1:STEP_SIZE:size[2])
            obstacle_r = rand(min_radius:STEP_SIZE:max_radius)
        end

        obstacle = Obstacle(obstacle_x, obstacle_y, obstacle_r)
        push!(obstacles, obstacle)
    end

    return obstacles
end


### transition function
function POMDPs.transition(env::DroneEnv, action::DroneAction)
    # scale actions
    accel = action.accel/10
    rotate = action.rotate/20

    # update heading angle
    theta_new = env.drone.theta + clamp(rotate, -env.max_rotation_rate, env.max_rotation_rate)
    theta_new = atan(sin(theta_new), cos(theta_new)) # limit to -π to π 

    # velocity update
    accel = clamp(accel, -env.max_acceleration, env.max_acceleration)
    v_new = clamp(env.drone.v + accel, -env.max_velocity, env.max_velocity)
    v_new += randn()*0.005 # add some randomness

    # update position
    x_new = env.drone.x + v_new*cos(theta_new)
    y_new = env.drone.y + v_new*sin(theta_new)
    x_new += randn()*0.05 # add some randomness
    y_new += randn()*0.05

    # update drone state in env
    new_state = DroneState(x_new, y_new, v_new, theta_new)
    env.drone = new_state
    
    if isterminal(env) != 0
        env.isterminal = true
    end

    return new_state, env.isterminal
end

function get_distance(v1, v2)
    return norm(v1 - v2)
end

### terminal condition handling
function isterminal(env::DroneEnv)
    distance_to_target = get_distance([env.drone.x, env.drone.y], [env.target.x, env.target.y])
    if distance_to_target <= env.target.r
        return 1
    end

    for i in 1:env.num_obstacles
        distance_to_obstacle = get_distance([env.drone.x, env.drone.y], [env.obstacles[i].x, env.obstacles[i].y])

        if distance_to_obstacle <= env.obstacles[i].r
            return 2
        end
    end

    # check for out of bounds
    if env.drone.x >= env.size[1] || env.drone.x <= 0 || env.drone.y >= env.size[2] || env.drone.y <= 0
        return 3
    end

    return 0
end

### reward function
function POMDPs.reward(env::DroneEnv)
    if isterminal(env) == 1
        return 100.0
    elseif isterminal(env) == 2
        return -100.0
    elseif isterminal(env) == 3
        return -100.0
    else
        r = get_distance([env.drone.x, env.drone.y], [env.target.x, env.target.y])
        return -r
    end
end

### discount
POMDPs.discount(env::DroneEnv) = env.discount


### generate next state sp, and reward r 
function gen(env::DroneEnv, action::DroneAction)
    sp, done = POMDPs.transition(env, action)
    r = POMDPs.reward(env)

    return sp, r, done
end

### reset env
function reset!(env::DroneEnv)
    env.size = env.size
    env.drone = env.initial_state
    env.target = env.target
    env.max_velocity = env.max_velocity
    env.max_acceleration = env.max_acceleration
    env.max_rotation_rate = env.max_rotation_rate
    env.discount = env.discount
    env.isterminal = false
    env.num_obstacles = env.num_obstacles
    env.obstacles = env.obstacles

    return 0
end


### policy stuff
function random_policy()
    accel = 2*rand()-1
    rotate = 0.1*(rand() - 0.5)
    return DroneAction(accel, rotate)
end

function heuristic_policy(env::DroneEnv)
    angle_to_target = atan(env.target.y - env.drone.y, env.target.x - env.drone.x)

    distance_to_target = get_distance([env.drone.x, env.drone.y], [env.target.x, env.target.y])

    if abs(angle_to_target-env.drone.theta) > 0.1
        accel = 0
    else
        accel = distance_to_target
    end
    rotate = angle_to_target - env.drone.theta

    return DroneAction(accel, rotate)
end


### plot target
function plot_target(p::Plots.Plot, env::DroneEnv; color::Symbol, label::String)
    n = 100
    rads = range(0, stop=2*π, length=n)
    x = env.target.x .+ env.target.r.*cos.(rads)
    y = env.target.y .+ env.target.r.*sin.(rads)
    plot!(p, x, y, label=label, color=color, legend=true)
end


### plot obstacles
function plot_obstacles(p::Plots.Plot, env::DroneEnv; color::Symbol, label::String)
    n = 100
    rads = range(0, stop=2*π, length=n)
    for i in 1:env.num_obstacles-1
        x = env.obstacles[i].x .+ env.obstacles[i].r.*cos.(rads)
        y = env.obstacles[i].y .+ env.obstacles[i].r.*sin.(rads)
        plot!(p, x, y, label="", color=color, legend=false)
    end
    x = env.obstacles[env.num_obstacles].x .+ env.obstacles[env.num_obstacles].r.*cos.(rads)
    y = env.obstacles[env.num_obstacles].y .+ env.obstacles[env.num_obstacles].r.*sin.(rads)
    plot!(p, x, y, label=label, color=color, legend=true)
end


### generate target manually
function set_target!(env::DroneEnv, target::Target)
    env.target = target
end

function get_state(drone::DroneState)
    return [drone.x, drone.y, drone.v, drone.theta]
end

# for plotting
function _rotate(x, y, angle)
    x_rotated = cos(angle)*x - sin(angle)*y
    y_rotated = sin(angle)*x + cos(angle)*y    

    return x_rotated, y_rotated
end

### plot 1 frame of environment
function plot_frame(env::DroneEnv; show=true, t=nothing)
    p = plot(size=(800, 800), xlim=(0, env.size[1]), ylim=(0, env.size[2]), legend=false)

    plot_target(p, env, label="Target Region", color=:red)
    plot_obstacles(p, env, label="Obstacles", color=:black)

    x = [0.6, 0, 0, 0.6]
    y = [0.0, 0.2, -0.2, 0.0]
    rotated_x, rotated_y = _rotate(x, y, env.drone.theta)

    plot!([env.drone.x], [env.drone.y], mark=Shape(rotated_x, rotated_y), markersize=75, color=:blue, label="Drone", legend=:topright)
    if !isnothing(t)
        annotate!(env.size[1]/2, env.size[2]-2, text("t=$t", :center, 12))
    end

    if show
        display(p)
    end
end

function create_animation(batch_env::Vector{DroneEnv})
    anim = @animate for i in eachindex(batch_env)
        plot_frame(batch_env[i], show=false, t=i)
    end

    return anim
end


end # module