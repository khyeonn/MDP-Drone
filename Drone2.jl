module Drone2


using POMDPs
using StaticArrays
using LinearAlgebra
using Random
using POMDPModelTools
using Plots


export DroneState, DroneAct, Circle, Env, DroneMDP, render

# State of a Drone.
struct DroneState <: FieldVector{3, Float32}
    x::Float32 # x location in meters
    y::Float32 # y location in meters
    theta::Float32 # orientation in radian
end

# Struct for a Drone action
struct DroneAct <: FieldVector{2, Float32}
    v::Float32     # meters per second
    omega::Float32 # theta dot (rad/s)
end


# Obstacle and Target area Struct
struct Circle
    x::Float32 # x location in meters
    y::Float32 # y location in meters
    r::Float32 # radius in meters
end

# Enviroment containing goal region and obstcles
struct Env
    goal::Circle
    obstacles::Vector{Circle}
end

# generate random obstacles
function generate_obstacles(size::SVector{2, SVector{2, Float32}}, num_obstacles::Int64, goal::Circle)
    obstacles = Vector{Circle}()
    min_radius= 2.5
    max_radius = 7.5
    for _ in 1:num_obstacles
        obstacle_x, obstacle_y = rand(size[1][1]:size[1][2]), rand(size[2][1]:size[2][2])
        obstacle_r = rand(min_radius:max_radius)

        ## if target and obstacle regions overlap, generate new obstacle
        while norm([goal.x, goal.y].- [obstacle_x, obstacle_y]) < goal.r + obstacle_r
            obstacle_x, obstacle_y = rand(size[1][1]:size[1][2]), rand(size[2][1]:size[2][2])
            obstacle_r = rand(min_radius:max_radius)
        end
        obstacle = Circle(obstacle_x, obstacle_y, obstacle_r)
        push!(obstacles, obstacle)
    end
    return obstacles
end

# Define the Drone MDP.
mutable struct DroneMDP <: MDP{DroneState, DroneAct}
    drone::DroneState
    init_state::DroneState
    v_max::Float32 # maximum velocity of Drone [m/s]
    om_max::Float32 # maximum turn-rate of Drone [rad/s]
    dt::Float32 # simulation time-step [s]
    pen::Float32 # penalty for crash or out of bounds
    time_pen::Float32 # penalty per time-step
    goal_reward::Float32 # reward for reaching goal
    discount::Float32
    size::SVector{2, SVector{2, Float32}}
    num_obstacles::Int64
    goal::Circle
    env::Env
    status::Int64 # indicator whether robot has reached goal state or crashed
end

# function init Drone MDP.
function DroneMDP(;
            drone = DroneState(25,25,0),
            init_state = DroneState(25,25,0),
            v_max=2.0, 
            om_max=1.0, 
            dt=0.5, 
            pen=-1000.0, 
            time_pen=-1.0, 
            goal_reward=100.0,
            discount=0.95,
            size=SVector(SVector(Float32(0.0), Float32(50.0)), SVector(Float32(0.0),Float32( 50.0))),
            status=0,
            num_obstacles=10)
            Random.seed!(1)
            goal= Circle(rand(size[1][1]:size[1][2]), rand(size[2][1]:size[2][2]), 5.0)
            env = Env(goal,generate_obstacles(size,num_obstacles,goal))
        return DroneMDP(drone,init_state,v_max,om_max,dt,pen,time_pen,goal_reward,discount,
                        size,num_obstacles,goal,env,status)
end

# transition Drone state given curent state and action
POMDPs.transition(m::DroneMDP,s::DroneState, a::DroneAct) = get_next_state(m,m.drone,a)

# terminal condition handling
function terminal(m::DroneMDP,x,y)
    distance_to_target = norm([x, y].- [m.env.goal.x, m.env.goal.y])
    if distance_to_target <= m.env.goal.r
        return 1
    end

    for i in 1:length(m.env.obstacles)
        distance_to_obstacle = norm([x, y].- [m.env.obstacles[i].x, m.env.obstacles[i].y])
        if distance_to_obstacle <= m.env.obstacles[i].r
            return -1
        end
    end

    # check for out of bounds
    if x >= m.size[1][2] || x <= m.size[1][1] || y >= m.size[2][2] || y <= m.size[2][1]
        return -1
    end

    return 0
end

# next statse function
function get_next_state(m::DroneMDP,s::DroneState, a::DroneAct)
    v, om = a
    v = clamp(v, -m.v_max, m.v_max)
    om = clamp(om, -m.om_max, m.om_max)

    x, y, th = s
    dt = m.dt

    # dynamics assume drone rotates and then translates
    next_th = mod( th + om*dt,2*pi)

    # make sure we arent going through a wall
    p0 = SVector(x, y)
    heading = SVector(cos(next_th), sin(next_th))
    des_step = v*dt
    pos = p0 + des_step*heading

    # define next state
    return DroneState(pos[1], pos[2], next_th)
end

# defines reward function R(s,a,s')

function POMDPs.reward(m::DroneMDP,s::DroneState,a::DroneAct)
    # penalty for each timestep elapsed
    cum_reward = m.time_pen

    # terminal rewards
    cum_reward += m.goal_reward*(m.status == 1)
    cum_reward += m.pen*(m.status == -1)

    return cum_reward  
end

# determine if a terminal state has been reached
POMDPs.isterminal(m::DroneMDP) = abs(m.status) > 0.0

# define discount factor
POMDPs.discount(m::DroneMDP) = m.discount

# generate next state sp, and reward r 
function POMDPs.gen(m::DroneMDP, action::DroneAct)
    sp = transition(m,m.drone,action)
    r = reward(m,m.drone,action)

    return sp, r
end

function act!(m::DroneMDP, action::DroneAct)
    sp = transition(m,m.drone,action)
    if m.status == 0
        next_status =  terminal(m,sp.x,sp.y)
        m.status =next_status
    end
    r = reward(m,m.drone,action)
    m.drone = sp
    done = isterminal(m)
    return sp, r, done
end


function reset!(m::DroneMDP)
    m.drone = m.init_state
    m.status = 0
    return m.drone
end

#Generate points for the circle
function plot_circle(x, y, r, fill_color=:none)
    theta = LinRange(0, 2π, 100)  # Angle range from 0 to 2π
    circle_x = x .+ r * cos.(theta)
    circle_y = y .+ r * sin.(theta)
    return plot!(circle_x, circle_y, aspect_ratio=1, legend=false, fillalpha=0.7, fillopacity=0.7, fill=true, fillcolor=fill_color)
end

##Render MDP 
function render(m::DroneMDP; show=true, t=nothing)
    s = m.drone
    p = plot(size=(600, 600), xlim=(m.size[1][1]-2, m.size[1][2]+2), ylim=(m.size[2][1]-2, m.size[2][2]+2), legend=false)
    xticks!(0:5:m.size[1][2]+1)
    yticks!(0:5:m.size[2][2]+1)
    for (i, obs) in enumerate(m.env.obstacles)
        plot_circle(obs.x, obs.y, obs.r,:red)
    end
    plot_circle(m.env.goal.x, m.env.goal.y, m.env.goal.r,:green)
    quiver!([s[1]], [s[2]], quiver=[(1*cos(s[3]), 1*sin(s[3]))], color=:blue, arrow=true, linewidth=2)
    
    plot!([m.size[1][1]+0.5, m.size[2][1]+0.5], [m.size[1][1]+0.5, m.size[2][2]+0.5], color=:black, linewidth=2)
    plot!([m.size[1][1]+0.5, m.size[2][2]+0.5], [m.size[1][1]+0.5, m.size[2][1]+0.5], color=:black, linewidth=2)
    plot!([m.size[1][1]+0.5, m.size[2][2]+0.5], [m.size[1][2]+0.5, m.size[2][2]+0.5], color=:black, linewidth=2)
    plot!([m.size[1][2]+0.5, m.size[2][2]+0.5], [m.size[1][2]+0.5, m.size[2][1]+0.5], color=:black, linewidth=2)
    if !isnothing(t)
        annotate!(m.size[1][2]/2, m.size[2][2]-2, text("t=$t", :center, 12))
    end

    if show
        display(p)
    end
end


end # module
