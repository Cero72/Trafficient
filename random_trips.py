import os
import sys
import random
import argparse

# # We need to import Python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare environment variable 'SUMO_HOME'")

try:
    import sumolib
except ImportError:
    sys.exit("Error: Failed to import sumolib. Make sure SUMO_HOME is set correctly.")

print(f"Current working directory: {os.getcwd()}")
print(f"SUMO_HOME is set to: {os.environ.get('SUMO_HOME', 'Not set')}")

def generate_random_trips(net_file, route_file, simulation_time, vehicle_count):
    # print(f"Starting random trip generation...")
    # print(f"Net file: {net_file}")
    # print(f"Route file: {route_file}")

    if not os.path.exists(net_file):
        sys.exit(f"Error: Net file '{net_file}' does not exist.")

    try:
        print("Reading net file...")
        net = sumolib.net.readNet(net_file)
        print("Net file read successfully.")
    except Exception as e:
        sys.exit(f"Error reading net file: {str(e)}")

    edges = net.getEdges()
    print(f"Found {len(edges)} edges in the network.")

    trips = []
    for i in range(vehicle_count):
        from_edge = random.choice(edges)
        to_edge = random.choice(edges)
        while to_edge == from_edge:
            to_edge = random.choice(edges)
        
        try:
            route = net.getShortestPath(from_edge, to_edge)[0]
            if route:
                route_edges = " ".join([e.getID() for e in route])
                depart_time = random.uniform(0, simulation_time)
                trips.append((depart_time, f'    <vehicle id="veh{i}" type="vtypeauto" depart="{depart_time:.2f}">\n        <route edges="{route_edges}"/>\n    </vehicle>\n'))
        except:
            print(f"Could not find a route from {from_edge.getID()} to {to_edge.getID()}")

    # Sort trips by departure time
    trips.sort(key=lambda x: x[0])

    try:
        print(f"Writing to route file: {route_file}")
        with open(route_file, "w") as f:
            f.write('<routes>\n')
            f.write('    <vType id="vtypeauto" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55.56" color="1,1,0"/>\n')
            for _, trip in trips:
                f.write(trip)
            f.write('</routes>\n')
        print(f"Route file written successfully.")
    except Exception as e:
        sys.exit(f"Error writing route file: {str(e)}")

    if os.path.exists(route_file):
        print(f"Verified: Route file '{route_file}' has been created.")
        print(f"File size: {os.path.getsize(route_file)} bytes")
    else:
        print(f"Error: Route file '{route_file}' was not created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random trips for SUMO simulation")
    parser.add_argument("--output", type=str, default="random_traffic.rou.xml", help="Output route file name")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(script_dir, "traditional_traffic.net.xml")
    route_file = os.path.join(script_dir, args.output)
    simulation_time = 3600  # 1 hour simulation
    vehicle_count = 3000  # Increased from 150 to 2000

    # print(f"Script directory: {script_dir}")
    # print(f"Net file path: {net_file}")
    # print(f"Route file path: {route_file}")

    generate_random_trips(net_file, route_file, simulation_time, vehicle_count)
    # print(f"Random trips generation process completed.")



