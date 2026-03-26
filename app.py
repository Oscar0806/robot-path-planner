import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathfinder import Grid, astar, create_warehouse_layout
 
st.set_page_config(page_title="Robot Path Planner",
                   page_icon="\U0001f916", layout="wide")
st.title("\U0001f916 Autonomous Robot Path Planning Simulator")
st.markdown("**A* pathfinding on a warehouse grid** \u2013 "
            "**Unmanned Systems concept**")
st.divider()
 
# ── SIDEBAR ──
st.sidebar.header("\u2699\uFE0F Settings")
grid_size = st.sidebar.slider("Grid size", 10, 40, 20)
layout_type = st.sidebar.selectbox("Layout",
    ["Warehouse (shelves)", "Random obstacles", "Empty"])
obstacle_density = st.sidebar.slider(
    "Obstacle density (%)", 0, 40, 20,
    help="For random layout only") if layout_type == "Random obstacles" else 20
 
start_x = st.sidebar.number_input("Start X", 0, grid_size-1, 1)
start_y = st.sidebar.number_input("Start Y", 0, grid_size-1, 1)
goal_x = st.sidebar.number_input("Goal X", 0, grid_size-1, grid_size-2)
goal_y = st.sidebar.number_input("Goal Y", 0, grid_size-1, grid_size-2)
 
# ── BUILD GRID ──
if layout_type == "Warehouse (shelves)":
    grid = create_warehouse_layout(grid_size, grid_size)
elif layout_type == "Random obstacles":
    grid = Grid(grid_size, grid_size)
    np.random.seed(42)
    n_obs = int(grid_size * grid_size * obstacle_density / 100)
    for _ in range(n_obs):
        ox, oy = np.random.randint(0, grid_size, 2)
        grid.add_obstacle(int(ox), int(oy))
    # Clear start and goal
    grid.grid[start_y][start_x] = 0
    grid.grid[goal_y][goal_x] = 0
else:
    grid = Grid(grid_size, grid_size)
 
# ── RUN A* ──
start = (int(start_x), int(start_y))
goal = (int(goal_x), int(goal_y))
path, explored = astar(grid, start, goal)
 
# ── KPIs ──
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Grid", f"{grid_size}x{grid_size}")
with c2: st.metric("Path Length", f"{len(path)} steps")
with c3: st.metric("Cells Explored", len(explored))
with c4:
    efficiency = (len(path)/len(explored)*100) if explored else 0
    st.metric("Efficiency", f"{efficiency:.0f}%")
 
if not path:
    st.error("\u274C No path found! Start or goal may be blocked.")
 
st.divider()
 
# ── VISUALIZATION ──
st.subheader("\U0001f5fa\uFE0F Grid World Visualization")
fig = go.Figure()
 
# Draw grid cells
for y in range(grid_size):
    for x in range(grid_size):
        if grid.grid[y][x] == 1:
            fig.add_shape(type="rect",
                x0=x, y0=y, x1=x+1, y1=y+1,
                fillcolor="#2C3E50", line_width=0.5,
                line_color="#1A252F")
 
# Draw explored cells (light blue)
for (ex, ey) in explored:
    if (ex, ey) not in path:
        fig.add_shape(type="rect",
            x0=ex, y0=ey, x1=ex+1, y1=ey+1,
            fillcolor="rgba(52,152,219,0.2)", line_width=0)
 
# Draw path (green)
if path:
    px = [p[0]+0.5 for p in path]
    py = [p[1]+0.5 for p in path]
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers",
        line=dict(color="#27AE60", width=3),
        marker=dict(size=4, color="#27AE60"),
        name="Optimal path"))
 
# Start and Goal markers
fig.add_trace(go.Scatter(
    x=[start[0]+0.5], y=[start[1]+0.5],
    mode="markers+text", text=["START"],
    textposition="top center",
    marker=dict(size=15, color="#3498DB", symbol="circle"),
    name="Start"))
fig.add_trace(go.Scatter(
    x=[goal[0]+0.5], y=[goal[1]+0.5],
    mode="markers+text", text=["GOAL"],
    textposition="top center",
    marker=dict(size=15, color="#E74C3C", symbol="star"),
    name="Goal"))
 
fig.update_layout(
    xaxis=dict(range=[0, grid_size], dtick=1, showgrid=True,
               gridcolor="rgba(0,0,0,0.1)"),
    yaxis=dict(range=[0, grid_size], dtick=1, showgrid=True,
               gridcolor="rgba(0,0,0,0.1)", scaleanchor="x"),
    height=600, template="plotly_white",
    legend=dict(orientation="h", y=-0.05),
    title=f"A* Path: {len(path)} steps, {len(explored)} cells explored")
st.plotly_chart(fig, use_container_width=True)
 
# ── PATH DETAILS ──
if path:
    st.subheader("\U0001f4cb Path Details")
    path_cost = 0
    for i in range(1, len(path)):
        dx = abs(path[i][0]-path[i-1][0])
        dy = abs(path[i][1]-path[i-1][1])
        path_cost += 1.414 if dx+dy == 2 else 1.0
    st.markdown(f"**Total path cost:** {path_cost:.1f} units")
    st.markdown(f"**Diagonal moves:** "
        f"{sum(1 for i in range(1,len(path)) if abs(path[i][0]-path[i-1][0])+abs(path[i][1]-path[i-1][1])==2)}")
    st.markdown(f"**Straight moves:** "
        f"{sum(1 for i in range(1,len(path)) if abs(path[i][0]-path[i-1][0])+abs(path[i][1]-path[i-1][1])==1)}")
 
st.divider()
st.caption("Robot Path Planning Simulator | Unmanned Systems Concept | "
           "Built by Oscar Vincent Dbritto")
