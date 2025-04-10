from ortools.sat.python import cp_model
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2

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

def optimize_bus_routes_cpsat(routes, passenger_demand, capacity=50, extra_buses=5):
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

# Bus stop coordinates (latitude, longitude)
bus_stops = {'Ambattur Estate': (13.10051, 80.1637), 'Ratinakanaru': (12.6842, 79.98333), 'Chengalpettu New BS': (12.69136, 79.98064), 'Chengalpettu Old BS': (12.69715, 79.97664), 'Mahindra City': (12.74371, 79.99251), 'Singaperumal Koil Signal': (12.76168, 80.00351), 'Ford BS': (12.78727, 80.015), 'Maraimalai Nagar BS': (12.79982, 80.0233), 'HP PB': (12.80286, 80.02538), 'Gurukulam': (12.815, 80.03268), 'Potheri BS': (12.82079, 80.03728), 'A2Z': (12.83714, 80.05204), 'Mambakkam': (12.82805, 80.16537), 'Peravallur BS': (13.11675, 80.23088), 'Venus (Gandhi Statue)': (13.11179, 80.239), 'Perambur Rly St.': (13.10861, 80.24235), 'Jamalia': (13.10692, 80.2476), 'Ottery': (13.09722, 80.25122), 'Porur (Kumar Sweets)': (13.03482, 80.15599), 'Saravana stores (shell PB)': (13.02178, 80.14535), 'Mugalaivakkam BS': (13.02128, 80.16065), 'Ramapuram BS': (13.02408, 80.17839), 'Sanitorium (GK Hotel)': (12.93815, 80.12928081), 'Perungalathur': (12.90655, 80.09684), 'Beach Station': (13.09302, 80.29224), 'MGR Janaki College': (13.01652, 80.2595), 'Adyar Depot(T.Exchange)': (12.99899, 80.25646), 'Thiruvanmiyur Post office': (12.98289, 80.25253), 'V.House': (13.05038, 80.2807), 'F.Shore Estate': (13.02469, 80.278), 'MRC Nagar': (13.01979, 80.26866), 'Wavin': (13.08898, 80.1752)}

# Define routes
routes = [['Ambattur Estate'], ['Ratinakanaru', 'Chengalpettu New BS', 'Chengalpettu Old BS', 'Mahindra City', 'Singaperumal Koil Signal', 'Ford BS', 'Maraimalai Nagar BS', 'HP PB', 'Gurukulam', 'Potheri BS', 'A2Z', 'Mambakkam',], ['Peravallur BS', 'Venus (Gandhi Statue)', 'Perambur Rly St.', 'Jamalia', 'Ottery'], ['Porur (Kumar Sweets)', 'Saravana stores (shell PB)'], ['College', 'Mugalaivakkam BS', 'Ramapuram BS', 'Sanitorium (GK Hotel)', 'Perungalathur',], ['Beach Station', 'MGR Janaki College', 'Adyar Depot(T.Exchange)', 'Thiruvanmiyur Post office'], ['Beach Station', 'V.House', 'F.Shore Estate', 'MRC Nagar', ], ['Wavin', 'Ambattur Estate', ]]

# Evening demand
passenger_demand_evening = {'Adyar Depot(T.Exchange)': 4, 'Ambattur Estate': 38, 'Beach Station': 31, 'Chengalpettu New BS': 1, 'Chengalpettu Old BS': 9, 'F.Shore Estate': 3, 'Gurukulam': 2, 'HP PB': 1, 'MGR Janaki College': 1, 'MRC Nagar': 13, 'Mahindra City': 1, 'Mambakkam': 3, 'Maraimalai Nagar BS': 7, 'Mugalaivakkam BS': 3, 'Ottery': 1, 'Peravallur BS': 33, 'Perungalathur': 20, 'Porur (Kumar Sweets)': 28, 'Potheri BS': 4, 'Ramapuram BS': 6, 'Ratinakanaru': 8, 'Sanitorium (GK Hotel)': 5, 'Saravana stores (shell PB)': 7, 'Singaperumal Koil Signal': 3, 'Thiruvanmiyur Post office': 8, 'Venus (Gandhi Statue)': 11, 'Wavin': 24}

# Afternoon demand
passenger_demand_afternoon = {'Ambattur Estate': 11, 'Beach Station': 3, 'Chengalpettu New BS': 2, 'Chengalpettu Old BS': 2, 'MRC Nagar': 1, 'Maraimalai Nagar BS': 1, 'Perungalathur': 2, 'Porur (Kumar Sweets)': 10, 'Ratinakanaru': 1, 'Sanitorium (GK Hotel)': 1, 'Saravana stores (shell PB)': 3, 'Wavin': 5}

# Max walking distance allowed (in kilometers)
max_walk_distance = 1.5
# Set bus capacity
bus_capacity = 100

# Run the optimization
best_evening_routes, best_afternoon_routes = find_best_evening_for_afternoon_cpsat(
    routes, passenger_demand_evening, passenger_demand_afternoon, bus_stops, max_walk_distance, bus_capacity
)

# Extract all bus stops from routes and bus_stops
all_stops = set(bus_stops.keys())
for route in routes:
    all_stops.update(route)

# Check for missing stops in evening and afternoon demand
missing_evening = set(passenger_demand_evening.keys()) - all_stops
missing_afternoon = set(passenger_demand_afternoon.keys()) - all_stops

print("Stops in Evening Demand but not in Stops/Routes:", missing_evening)
print("Stops in Afternoon Demand but not in Stops/Routes:", missing_afternoon)


# Print results
print("Best Evening Routes Selected:", best_evening_routes)
if best_afternoon_routes is not None:
    print("Best Afternoon Routes Selected (from unused evening routes):", best_afternoon_routes)
else:
    print("No feasible afternoon solution found with remaining routes.")