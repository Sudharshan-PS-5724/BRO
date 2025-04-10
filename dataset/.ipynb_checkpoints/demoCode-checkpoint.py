from ortools.sat.python import cp_model
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2
from demandDistribution import create_location_coordinates, create_routes_list,create_demand_distribution
from glob import glob

def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two latitude/longitude coordinates.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius_of_earth = 6371  # Radius of Earth in kilometers
    distance = radius_of_earth * c

    return distance

def calculate_distances(bus_stops):
    """
    Precompute the pairwise distances between all bus stops using Haversine distance.
    """
    stop_names = list(bus_stops.keys())
    dist_matrix = {stop: {} for stop in stop_names}
    
    for i, j in combinations(stop_names, 2):
        distance = haversine_distance(bus_stops[i], bus_stops[j])
        dist_matrix[i][j] = distance
        dist_matrix[j][i] = distance
    #fill the diagonal with 0.
    for stop in stop_names:
        dist_matrix[stop][stop] = 0

    return dist_matrix

def optimize_bus_routes_cpsat(routes, passenger_demand, capacity=50, extra_buses=0):
    """
    Solve the evening bus selection problem optimally using CPSAT,
    allowing alternative solutions with up to `extra_buses` more than the optimal.
    """
    model = cp_model.CpModel()
    x = {r: model.NewBoolVar(f'x_{r}') for r in range(len(routes))}

    # Ensure all students at each stop are accommodated
# Ensure all students at each stop are accommodated
    for stop, demand in passenger_demand.items():
        model.Add(
            sum(x[r] for r in range(len(routes)) if stop in routes[r]) >= (1 if demand > 0 else 0)
        )

    # Capacity constraints
    for r in range(len(routes)):
        total_passengers = sum(passenger_demand.get(stop, 0) for stop in routes[r])
        model.Add(x[r] * total_passengers <= capacity)

    # Minimize the number of buses used (first solve for optimal)
    model.Minimize(sum(x[r] for r in range(len(routes))))
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        B_opt = sum(solver.Value(x[r]) for r in range(len(routes)))  # Optimal buses used
        selected_routes = [r for r in range(len(routes)) if solver.Value(x[r]) > 0.5]

        # Now, find alternative solutions allowing up to B_opt + extra_buses
        alternative_solutions = []
        new_model = cp_model.CpModel()
        new_x = {r: new_model.NewBoolVar(f'new_x_{r}') for r in range(len(routes))}

        # Copy constraints from the original model
        for stop, demand in passenger_demand.items():
            new_model.Add(
                sum(new_x[r] * min(capacity, passenger_demand.get(stop, 0))
                    for r in range(len(routes)) if stop in routes[r]) >= demand
            )

        for r in range(len(routes)):
            total_passengers = sum(passenger_demand.get(stop, 0) for stop in routes[r])
            new_model.Add(new_x[r] * total_passengers <= capacity)

        # Allow solutions with at most B_opt + extra_buses
        new_model.Add(sum(new_x[r] for r in range(len(routes))) <= B_opt + extra_buses)

        # Search for alternative solutions
        class SolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                super().__init__()
                self.solutions = []

            def OnSolutionCallback(self):
                solution = [r for r in range(len(routes)) if self.Value(new_x[r]) > 0.5]
                self.solutions.append(solution)

        collector = SolutionCollector()
        new_solver = cp_model.CpSolver()
        new_solver.SearchForAllSolutions(new_model, collector)

        return selected_routes, collector.solutions

    else:
        return None, None


def optimize_afternoon_routes_cpsat(remaining_routes, afternoon_demand, bus_stops, max_walk_distance=0.5, capacity=50):
    """
    Solve the afternoon bus problem while minimizing total walking distance,
    ensuring no student walks more than max_walk_distance.
    """
    if not remaining_routes or not afternoon_demand:
        return None, float('inf')
    
    # Calculate distances between all stops
    distances = calculate_distances(bus_stops)
    
    # Create the CP-SAT model
    model = cp_model.CpModel()
    
    # Decision variables for route selection
    x = {r: model.NewBoolVar(f'x_{r}') for r in range(len(remaining_routes))}
    
    # Decision variables for stop assignment - which stop each demand point uses
    y = {}
    for demand_stop in afternoon_demand:
        for r in range(len(remaining_routes)):
            for route_stop in remaining_routes[r]:
                # Can only assign if distance is within max_walk_distance
                if distances[demand_stop][route_stop] <= max_walk_distance:
                    y[(demand_stop, r, route_stop)] = model.NewBoolVar(f'y_{demand_stop}_{r}_{route_stop}')
    
    # Each demand point must be assigned to exactly one route stop
    for demand_stop in afternoon_demand:
        assign_vars = [
            y.get((demand_stop, r, route_stop), 0)
            for r in range(len(remaining_routes))
            for route_stop in remaining_routes[r]
            if (demand_stop, r, route_stop) in y
        ]
        if assign_vars:  # Only add constraint if there are possible assignments
            model.Add(sum(assign_vars) == 1)
        else:
            # If no possible assignments, this problem is infeasible
            return None, float('inf')
    
    # A demand point can only be assigned to a stop on a selected route
    for demand_stop in afternoon_demand:
        for r in range(len(remaining_routes)):
            for route_stop in remaining_routes[r]:
                if (demand_stop, r, route_stop) in y:
                    model.Add(y[(demand_stop, r, route_stop)] <= x[r])
    
    # Ensure bus capacity is not exceeded
    for r in range(len(remaining_routes)):
        assigned_demand = sum(
            y.get((demand_stop, r, route_stop), 0) * afternoon_demand[demand_stop]
            for demand_stop in afternoon_demand
            for route_stop in remaining_routes[r]
            if (demand_stop, r, route_stop) in y
        )
        model.Add(assigned_demand <= capacity * x[r])
    
    # Add constraints to ensure routes are only selected if they're used
    for r in range(len(remaining_routes)):
        route_usage = sum(
            y.get((demand_stop, r, route_stop), 0)
            for demand_stop in afternoon_demand
            for route_stop in remaining_routes[r]
            if (demand_stop, r, route_stop) in y
        )
        # If no demand points use this route, don't select it
        model.Add(route_usage > 0).OnlyEnforceIf(x[r])
    
    # Objective: Minimize number of routes first, then walking distance
    route_weight = 10000  # Large weight for route minimization
    
    total_walking_distance = sum(
        y.get((demand_stop, r, route_stop), 0) * distances[demand_stop][route_stop] * afternoon_demand[demand_stop]
        for demand_stop in afternoon_demand
        for r in range(len(remaining_routes))
        for route_stop in remaining_routes[r]
        if (demand_stop, r, route_stop) in y
    )
    
    # Combined objective: Minimize routes first, then walking distance
    model.Minimize(route_weight * sum(x[r] for r in range(len(remaining_routes))) + total_walking_distance)
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_routes = [r for r in range(len(remaining_routes)) if solver.Value(x[r]) > 0.5]
        
        # Calculate the actual total walking distance
        walk_distance = sum(
            solver.Value(y.get((demand_stop, r, route_stop), 0)) * 
            distances[demand_stop][route_stop] * 
            afternoon_demand[demand_stop]
            for demand_stop in afternoon_demand
            for r in range(len(remaining_routes))
            for route_stop in remaining_routes[r]
            if (demand_stop, r, route_stop) in y
        )
        
        return [remaining_routes[r] for r in selected_routes], walk_distance
    else:
        return None, float('inf')

def find_best_evening_for_afternoon_cpsat(routes, passenger_demand_evening, passenger_demand_afternoon, bus_stops, max_walk_distance=0.5, capacity=50):
    """
    Finds the best evening bus selection that maximizes afternoon feasibility.
    """
    best_evening_routes, alternative_solutions = optimize_bus_routes_cpsat(routes, passenger_demand_evening, capacity)
    if not best_evening_routes:
        return None, None

    best_afternoon_routes = None
    min_total_walk_distance = float('inf')

    for evening_solution in alternative_solutions:
        remaining_routes = [routes[r] for r in range(len(routes)) if r not in evening_solution]
        afternoon_routes, total_walk_distance = optimize_afternoon_routes_cpsat(remaining_routes, passenger_demand_afternoon, bus_stops, max_walk_distance, capacity)

        if afternoon_routes and total_walk_distance < min_total_walk_distance:
            min_total_walk_distance = total_walk_distance
            best_evening_routes = evening_solution
            best_afternoon_routes = afternoon_routes

    return best_evening_routes, best_afternoon_routes

# =======================
# *EXAMPLE USAGE*
# =======================

print("PROGRAM BEGINS...", end = "\n")

# Creating the location coordinates dictionary
coordinates_file_path = "dataset\\allRoutesLatLong.xlsx"
bus_stops = create_location_coordinates(coordinates_file_path)

# Creating a list of routes
route_list_file_path = "dataset\\routesList.xlsx"
route_list = create_routes_list(route_list_file_path)

# Creating evening demand
excel_folder_path = "dataset/"  # Make sure this points to your folder
excel_files = glob(excel_folder_path + "*.xlsx")

# Evening demand filters
evening_filters = {
    'YEAR': ["second","third", "fourth"],  
    'COLLEGE': ['SSN', "SNU", "faculty"]  
}

passenger_demand_evening = create_demand_distribution(excel_files, evening_filters)

# Creating afternoon demand
# Afternoon demand filters
afternoon_filters = {
    'YEAR': ["first"],  
    'COLLEGE': ["SSN"]  
}

passenger_demand_afternoon = create_demand_distribution(excel_files, afternoon_filters)

# Max walking distance allowed (in kilometers)
max_walk_distance = 1.5
# Set bus capacity
bus_capacity = 100


# Run the optimization
best_evening_routes, best_afternoon_routes = find_best_evening_for_afternoon_cpsat(
    route_list, passenger_demand_evening, passenger_demand_afternoon, bus_stops, max_walk_distance, bus_capacity
)

# Print results
print("Best Evening Routes Selected:", best_evening_routes)
if best_afternoon_routes is not None:
    print("Best Afternoon Routes Selected (from unused evening routes):", best_afternoon_routes)
else:
    print("No feasible afternoon solution found with remaining routes.") # Need to give all remaining routes
