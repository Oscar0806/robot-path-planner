import numpy as np
import heapq
 
class Grid:
    """2D grid world for robot navigation."""
    def __init__(self, width=20, height=20):
        self.w = width
        self.h = height
        self.grid = np.zeros((height, width), dtype=int)
        # 0=free, 1=obstacle
    
    def add_obstacle(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.grid[y][x] = 1
    
    def add_rect_obstacle(self, x, y, w, h):
        for dx in range(w):
            for dy in range(h):
                self.add_obstacle(x+dx, y+dy)
    
    def is_free(self, x, y):
        return (0 <= x < self.w and 0 <= y < self.h
                and self.grid[y][x] == 0)
 
def heuristic(a, b):
    """Manhattan distance."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])
 
def astar(grid, start, goal):
    """
    A* pathfinding algorithm.
    Returns: path (list of (x,y)), explored (set of visited cells)
    """
    if not grid.is_free(start[0], start[1]):
        return [], set()
    if not grid.is_free(goal[0], goal[1]):
        return [], set()
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    explored = set()
    
    # 8-directional movement
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], explored
        
        explored.add(current)
        
        for dx, dy in neighbors:
            nx, ny = current[0]+dx, current[1]+dy
            if not grid.is_free(nx, ny):
                continue
            neighbor = (nx, ny)
            
            # Diagonal moves cost sqrt(2)
            move_cost = 1.414 if abs(dx)+abs(dy)==2 else 1.0
            tentative_g = g_score[current] + move_cost
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    
    return [], explored  # No path found
 
def create_warehouse_layout(width=20, height=20):
    """Pre-built warehouse with shelf obstacles."""
    g = Grid(width, height)
    # Shelf rows
    for row_y in [3, 7, 11, 15]:
        for x in range(3, width-3):
            if x % 5 != 0:  # Leave aisles every 5 cells
                g.add_obstacle(x, row_y)
                g.add_obstacle(x, row_y+1)
    return g
 
if __name__ == "__main__":
    g = create_warehouse_layout()
    path, explored = astar(g, (1, 1), (18, 18))
    print(f"Path length: {len(path)} steps")
    print(f"Cells explored: {len(explored)}")
    print(f"Path: {path[:5]}...{path[-3:]}")
    if not path:
        print("No path found!")
