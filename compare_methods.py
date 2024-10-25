import os
import numpy as np
from rl_traffic_control import train, test_saved_model
from traditional_traffic_control import test_traditional_control

# Add these constants at the top
SIMULATION_DURATION = 3600  # 1 hour in seconds
MEASUREMENT_INTERVAL = 5    # 5 seconds
NUM_EPISODES = 10
TRAFFIC_DENSITY = {
    'low': 0.1,
    'medium': 0.3,
    'high': 0.5
}

def train_and_test_rl_model(num_episodes=1000):
    try:
        # Train the RL model
        trained_agent, rewards, waiting_times, avg_speeds = train(episodes=num_episodes)
        
        # Convert speeds to km/h and cap at 50 km/h
        avg_speeds = np.minimum(np.array(avg_speeds) * 3.6, 50.0)
        
        # Use the last 10 episodes for metrics
        last_10_waiting = waiting_times[-20:]
        last_10_speeds = avg_speeds[-20:]
        
        return np.mean(last_10_waiting), np.mean(last_10_speeds)
    except Exception as e:
        print(f"Error during RL training/testing: {e}")
        return None, None

def format_metrics(waiting_time, speed):
    return f"Avg Waiting Time: {waiting_time:.2f} seconds, Avg Speed: {speed:.2f} km/h"

def test_saved_rl_model(model_path, num_episodes=5):
    """Test the saved RL model and return metrics"""
    try:
        # Load and test the saved model
        metrics = test_saved_model(model_path, num_episodes)
        
        # Get average waiting times and speeds from the metrics dictionary
        avg_waiting_time = np.mean(metrics['waiting_times'])
        avg_speed = np.mean(metrics['speeds'])
        
        # Convert speeds to km/h and cap at 50 km/h
        avg_speed = min(avg_speed * 3.6, 50.0)
        
        return avg_waiting_time, avg_speed
    except Exception as e:
        print(f"Error during RL testing: {e}")
        return None, None

def run_comparison(model_path, num_episodes=5, traffic_density='medium'):
    """Run comparison using saved RL model"""
    env_params = {
        'simulation_duration': SIMULATION_DURATION,
        'measurement_interval': MEASUREMENT_INTERVAL,
        'traffic_density': TRAFFIC_DENSITY[traffic_density]
    }
    
    # Test saved RL model
    print(f"\nTesting saved RL model with {traffic_density} traffic density...")
    rl_waiting_time, rl_avg_speed = test_saved_rl_model(model_path, num_episodes)

    if rl_waiting_time is not None and rl_avg_speed is not None:
        # Test traditional control
        print(f"\nTesting traditional control with {traffic_density} traffic density...")
        trad_waiting_time, trad_avg_speed = test_traditional_control(num_episodes=num_episodes)
        
        # Cap traditional speeds at 50 km/h
        trad_avg_speed = min(trad_avg_speed, 50.0)

        print("\nComparison Results:")
        print(f"RL Method - {format_metrics(rl_waiting_time, rl_avg_speed)}")
        print(f"Traditional Method - {format_metrics(trad_waiting_time, trad_avg_speed)}")

        # Calculate improvements
        waiting_improvement = ((trad_waiting_time - rl_waiting_time) / max(trad_waiting_time, 0.1)) * 100
        speed_improvement = ((rl_avg_speed - trad_avg_speed) / max(trad_avg_speed, 0.1)) * 100

        print("\nImprovements:")
        print(f"Waiting Time: {waiting_improvement:+.2f}% ({'better' if waiting_improvement > 0 else 'worse'})")
        print(f"Average Speed: {speed_improvement:+.2f}% ({'better' if speed_improvement > 0 else 'worse'})")
    else:
        print("RL model testing did not complete successfully.")

if __name__ == "__main__":
    model_path = 'trained_dqn_agent_episodes_100.pth'
    run_comparison(model_path, num_episodes=5, traffic_density='medium')
