import os
import numpy as np
import traci
import xml.etree.ElementTree as ET
from rl_traffic_control import TrafficEnvironment, generate_new_random_traffic

# Define the function to test traditional traffic control
def test_traditional_control(num_episodes=10):
    # Initialize metrics storage
    metrics = {
        'waiting_times': [],
        'speeds': [],
        'vehicles': [],
        'queue_lengths': []
    }
    
    # Fixed-time traffic signal plan
    phase_durations = [30, 30, 30, 30]  # Duration for each phase in seconds
    phase_sequence = [0, 2, 4, 6]  # Main phases for N-S and E-W movements
    
    for episode in range(num_episodes):
        try:
            route_file = generate_new_random_traffic()
            env = TrafficEnvironment(os.path.join(os.path.dirname(__file__), "traditional_traffic.sumo.cfg"))
            state = env.reset(route_file)
            
            current_phase_idx = 0
            time_in_phase = 0
            
            episode_metrics = {
                'waiting_times': [],
                'speeds': [],
                'vehicles': [],
                'queue_lengths': []
            }
            
            # Simulate traditional fixed-time control
            for step in range(3600):  # 1-hour simulation
                try:
                    # Update phase if duration exceeded
                    if time_in_phase >= phase_durations[current_phase_idx]:
                        current_phase_idx = (current_phase_idx + 1) % len(phase_sequence)
                        time_in_phase = 0
                    
                    action = phase_sequence[current_phase_idx]
                    
                    # Step simulation
                    next_state, reward, done = env.step(action)
                    time_in_phase += 1
                    
                    # Collect metrics only if TRACI is still connected
                    if traci.isLoaded():
                        waiting_time, avg_speed, vehicle_count = env.get_performance_metrics()
                        queue_lengths = env.get_queue_lengths()  # New helper method
                        avg_queue_length = sum(queue_lengths.values()) / len(queue_lengths) if queue_lengths else 0
                        
                        if vehicle_count > 0:  # Only record metrics when vehicles are present
                            episode_metrics['waiting_times'].append(waiting_time)
                            episode_metrics['speeds'].append(avg_speed)
                            episode_metrics['vehicles'].append(vehicle_count)
                            episode_metrics['queue_lengths'].append(avg_queue_length)
                    
                    if done:
                        break
                        
                except traci.exceptions.FatalTraCIError:
                    break
            
            # Calculate episode averages
            if episode_metrics['vehicles']:  # If we had any vehicles
                for metric in ['waiting_times', 'speeds', 'vehicles', 'queue_lengths']:
                    if episode_metrics[metric]:
                        metrics[metric].append(np.mean(episode_metrics[metric]))
            
            print(f"Episode {episode + 1}/{num_episodes} completed")
            
            # Ensure TRACI connection is closed
            if traci.isLoaded():
                traci.close()
                
        except Exception as e:
            print(f"Error in episode {episode + 1}: {str(e)}")
            if traci.isLoaded():
                traci.close()
            continue

    # Calculate overall averages
    avg_metrics = {}
    for metric in metrics:
        if metrics[metric]:
            avg_metrics[metric] = np.mean(metrics[metric])
        else:
            avg_metrics[metric] = 0.0

    print("\nTraditional Control Results:")
    print(f"Average Waiting Time: {avg_metrics['waiting_times']:.2f} seconds")
    print(f"Average Speed: {avg_metrics['speeds']:.2f} m/s")
    print(f"Average Queue Length: {avg_metrics['queue_lengths']:.2f} vehicles")

    return (avg_metrics['waiting_times'], 
            avg_metrics['speeds'], 
            )

# Example usage
if __name__ == "__main__":
    num_episodes = 10
    avg_waiting_time, avg_speed, avg_queue_length = test_traditional_control(num_episodes)
    print(f"\nFinal Averages:")
    print(f"Waiting Time: {avg_waiting_time:.2f} seconds")
    print(f"Speed: {avg_speed:.2f} m/s")
    print(f"Queue Length: {avg_queue_length:.2f} vehicles")
