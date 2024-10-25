import os
import sys
import traci
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import warnings
import math
import time


warnings.filterwarnings("ignore")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# We need to import Python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO Configuration
sumocfg_file = os.path.join(script_dir, "traditional_traffic.sumo.cfg")
sumo_cmd = ["sumo", "-c", sumocfg_file]  # Changed to "sumo" for faster training

print(f"SUMO configuration file path: {sumocfg_file}")
print(f"Current working directory: {os.getcwd()}")

# Check if the configuration file exists
if not os.path.exists(sumocfg_file):
    raise FileNotFoundError(f"SUMO configuration file not found: {sumocfg_file}")

def generate_new_random_traffic():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    random_trips_script = os.path.join(script_dir, "random_trips.py")
    output_file = os.path.join(script_dir, "random_traffic.rou.xml")
    
    subprocess.run(["python", random_trips_script, "--output", output_file], check=True)
    return output_file

# DQN-specific imports and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class SimpleReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # Use deque for efficient FIFO operations

    def push(self, *args):
        self.memory.append(Transition(*args))  # Store transitions

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # Randomly sample transitions

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Define network layers without batch normalization
        self.layer1 = nn.Linear(42, 256)  # Updated input dimension to 42
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, n_actions)
        
        # Use layer normalization instead of batch normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.ln1(self.layer1(x)))
        x = torch.relu(self.ln2(self.layer2(x)))
        x = torch.relu(self.ln3(self.layer3(x)))
        return self.layer4(x)

class TrafficEnvironment:
    def __init__(self, sumocfg_file):
        self.sumocfg_file = sumocfg_file
        self.junction_id = "junction"
        self.phases = [0, 1, 2, 3, 4, 5, 6, 7]
        
        # Updated traffic control parameters
        self.measurement_interval = 5  # Reduced from 30 to 5 seconds
        self.min_green_time = 15      # Minimum green time in seconds
        self.max_red_time = 90        # Maximum red time for any approach
        self.phase_durations = {phase: 0 for phase in range(8)}
        self.last_phase_change = 0
        self.last_phase = 0
        
        # Rest of initialization remains the same
        self.current_phase = 0
        self.max_steps = 3600
        self.connection_closed = False
        self.sumo_cmd = ["sumo", "-c", sumocfg_file]
        self.steps = 0
        
        # Monitored edges
        self.monitored_edges = [
            "end1_junction", "end2_junction", "end3_junction", "end4_junction",
            "junction_end1", "junction_end2", "junction_end3", "junction_end4"
        ]
        
        # Normalization constants
        self.max_vehicles = 20
        self.max_waiting_time = 300
        self.max_speed = 13.89  # 50 km/h in m/s
        
        # Metrics tracking
        self.episode_metrics = {
            'waiting_times': [],
            'speeds': [],
            'vehicle_counts': [],
            'queue_lengths': []
        }
        self.speed_history = []

    def _check_queues(self):
        """Monitor and manage queue buildup"""
        queue_lengths = {}
        critical_queues = []
        
        for edge in self.monitored_edges:
            # Get halting vehicles (speed < 0.1 m/s)
            halting = traci.edge.getLastStepHaltingNumber(edge)
            queue_lengths[edge] = halting
            
            # Check for critical queue length (more than 5 vehicles)
            if halting > 5:
                critical_queues.append(edge)
        
        return queue_lengths, critical_queues

    def _get_state(self):
        """Get the current state of the traffic environment"""
        try:
            state = np.zeros(42, dtype=np.float32)  # Updated size for enhanced state
            
            if not traci.isLoaded():
                return torch.tensor(state, dtype=torch.float32, device=device)
                
            queue_lengths, critical_queues = self._check_queues()
            idx = 0
            
            for edge in self.monitored_edges:
                # Original metrics
                waiting_time = traci.edge.getWaitingTime(edge)
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                mean_speed = traci.edge.getLastStepMeanSpeed(edge)
                
                # Queue information
                queue_length = queue_lengths[edge]
                is_critical = 1.0 if edge in critical_queues else 0.0
                
                # Normalize and add to state
                state[idx:idx+5] = [
                    min(vehicle_count / self.max_vehicles, 1.0),
                    min(waiting_time / self.max_waiting_time, 1.0),
                    min(mean_speed / self.max_speed, 1.0),
                    min(queue_length / 10.0, 1.0),
                    is_critical
                ]
                idx += 5
            
            # Add traffic light and simulation info
            state[40] = self.current_phase / len(self.phases)
            state[41] = self.steps / self.max_steps
            
            return torch.tensor(state, dtype=torch.float32, device=device)
            
        except traci.exceptions.FatalTraCIError:
            self.connection_closed = True
            return torch.zeros(42, device=device)

    def _get_reward(self):
        if self.connection_closed:
            return 0
            
        try:
            total_reward = 0
            edge_rewards = {}
            
            for edge in self.monitored_edges:
                # Get edge metrics
                waiting_time = traci.edge.getWaitingTime(edge)
                vehicles = traci.edge.getLastStepVehicleNumber(edge)
                mean_speed = traci.edge.getLastStepMeanSpeed(edge)
                
                if vehicles > 0:
                    # Progressive waiting time penalty
                    waiting_penalty = math.exp(min(waiting_time / 30.0, 2.0)) - 1
                    
                    # Speed reward component
                    speed_reward = mean_speed / self.max_speed
                    
                    # Throughput reward
                    flow_rate = len(traci.edge.getLastStepVehicleIDs(edge))
                    throughput_reward = min(flow_rate / 10.0, 1.0)
                    
                    # Combined edge reward
                    edge_rewards[edge] = (
                        -waiting_penalty * 0.5 +
                        speed_reward * 0.3 +
                        throughput_reward * 0.2
                    )
                    
                    total_reward += edge_rewards[edge]
            
            # Phase change penalty
            if self.current_phase != self.last_phase:
                phase_penalty = 5.0
                total_reward -= phase_penalty
            
            # Bonus for balanced flow
            if edge_rewards:
                variance = np.var(list(edge_rewards.values()))
                balance_bonus = math.exp(-variance) * 10
                total_reward += balance_bonus
            
            return float(total_reward)
            
        except traci.exceptions.FatalTraCIError:
            self.connection_closed = True
            return 0

    def step(self, action):
        if self.connection_closed:
            return self._get_state(), 0, True

        current_time = self.steps * self.measurement_interval
        
        # Check minimum green time
        if (current_time - self.last_phase_change) >= self.min_green_time:
            if action != self.current_phase:
                self.last_phase = self.current_phase
                self.last_phase_change = current_time
                self.current_phase = action
                traci.trafficlight.setPhase(self.junction_id, self.current_phase)
        
        # Simulate with smaller steps
        for _ in range(self.measurement_interval):
            self._update_metrics()  # Use the new method
            traci.simulationStep()
            self.steps += 1
        
        next_state = self._get_state()
        reward = self._get_reward()
        done = self.steps >= self.max_steps or traci.simulation.getMinExpectedNumber() <= 0
        
        return next_state, reward, done

    def reset(self, route_file):
        if traci.isLoaded():
            traci.close()
            
        self.update_sumocfg(route_file)
        traci.start(self.sumo_cmd)
        self.current_phase = 0
        self.steps = 0
        self.last_action_time = 0
        self.current_phase_duration = 0
        traci.trafficlight.setPhase(self.junction_id, self.current_phase)
        self.connection_closed = False
        
        # Reset all tracking variables
        self.speed_history = []
        self.episode_metrics = {
            'waiting_times': [],
            'speeds': [],
            'vehicle_counts': [],
            'queue_lengths': []
        }
        
        return self._get_state()


    def update_sumocfg(self, route_file):
        tree = ET.parse(self.sumocfg_file)
        root = tree.getroot()
        for route_files in root.iter('route-files'):
            route_files.set('value', route_file)
        tree.write(self.sumocfg_file)

    def step(self, action):
        if self.connection_closed:
            return self._get_state(), 0, True

        self.current_phase = action
        traci.trafficlight.setPhase(self.junction_id, self.current_phase)

        total_waiting_time = 0
        # Simulate and collect metrics
        for _ in range(30):
            self._update_metrics()
            waiting_time = sum(traci.edge.getWaitingTime(edge) for edge in self.monitored_edges)
            total_waiting_time += waiting_time
            traci.simulationStep()
        self.steps += 30

        new_state = self._get_state()
        reward = self._get_reward()
        
        # Early stopping conditions
        high_waiting_time = total_waiting_time > 3000  # Stop if waiting time is too high
        done = (self.connection_closed or 
                traci.simulation.getMinExpectedNumber() <= 0 or 
                self.steps >= self.max_steps or
                high_waiting_time)

        if done and not self.connection_closed:
            self._update_metrics()
            traci.close()
            self.connection_closed = True

        return new_state, reward, done

    def get_performance_metrics(self):
        """Return average metrics over the episode"""
        if not self.episode_metrics['vehicle_counts']:  # If no data collected
            return 0, 0, 0
        
        try:
            avg_waiting_time = (sum(self.episode_metrics['waiting_times']) / 
                            len(self.episode_metrics['waiting_times']))
        except ZeroDivisionError:
            avg_waiting_time = 0
            
        try:
            avg_speed = np.mean(self.speed_history) if self.speed_history else 0
        except:
            avg_speed = 0
            
        try:
            avg_vehicles = (sum(self.episode_metrics['vehicle_counts']) / 
                        len(self.episode_metrics['vehicle_counts']))
        except ZeroDivisionError:
            avg_vehicles = 0
        
        return avg_waiting_time, avg_speed, avg_vehicles

    def step(self, action):
        if self.connection_closed:
            return self._get_state(), 0, True

        self.current_phase = action
        traci.trafficlight.setPhase(self.junction_id, self.current_phase)

        total_waiting_time = 0
        step_speeds = []
        
        # Simulate and collect metrics
        for _ in range(self.measurement_interval):
            self._update_metrics()
            waiting_time = sum(traci.edge.getWaitingTime(edge) for edge in self.monitored_edges)
            total_waiting_time += waiting_time
            
            # Measure speed every step
            current_speed = self._measure_traffic_speed()
            if current_speed > 0:
                step_speeds.append(current_speed)
                self.speed_history.append(current_speed)
                
            traci.simulationStep()
            
        self.steps += self.measurement_interval

        new_state = self._get_state()
        reward = self._get_reward()
        
        # Early stopping conditions
        high_waiting_time = total_waiting_time > 3000
        done = (self.connection_closed or 
                traci.simulation.getMinExpectedNumber() <= 0 or 
                self.steps >= self.max_steps or
                high_waiting_time)

        if done and not self.connection_closed:
            self._update_metrics()
            traci.close()
            self.connection_closed = True

        return new_state, reward, done

    def _update_metrics(self):
        """Update traffic metrics for the current simulation step"""
        try:
            # Update waiting times
            waiting_time = sum(traci.edge.getWaitingTime(edge) for edge in self.monitored_edges)
            
            # Update speeds
            current_speed = self._measure_traffic_speed()
            if current_speed > 0:
                self.speed_history.append(current_speed)
            
            # Update vehicle counts
            vehicle_count = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in self.monitored_edges)
            
            # Update queue lengths
            queue_lengths, _ = self._check_queues()
            total_queue_length = sum(queue_lengths.values())
            
            # Only store metrics if there are vehicles in the network
            if vehicle_count > 0:
                self.episode_metrics['waiting_times'].append(waiting_time)
                self.episode_metrics['speeds'].append(current_speed)
                self.episode_metrics['vehicle_counts'].append(vehicle_count)
                self.episode_metrics['queue_lengths'].append(total_queue_length)
                
        except traci.exceptions.FatalTraCIError:
            self.connection_closed = True

    def _measure_traffic_speed(self):
        """Measure actual traffic speed considering all vehicles"""
        total_speed = 0
        total_vehicles = 0
        speeds = []
        
        for edge in self.monitored_edges:
            vehicle_ids = traci.edge.getLastStepVehicleIDs(edge)
            for vid in vehicle_ids:
                speed = traci.vehicle.getSpeed(vid)
                if speed > 0:  # Only count moving vehicles
                    speed_kmh = min(speed * 3.6, 50.0)  # Cap at 50 km/h
                    speeds.append(speed_kmh)
                    total_speed += speed_kmh
                    total_vehicles += 1
        
        if total_vehicles == 0:
            return 0
            
        return total_speed / total_vehicles

    def get_queue_lengths(self):
        """Safely get queue lengths for all monitored edges"""
        queue_lengths = {}
        try:
            if traci.isLoaded():
                for edge in self.monitored_edges:
                    queue_lengths[edge] = traci.edge.getLastStepHaltingNumber(edge)
        except traci.exceptions.FatalTraCIError:
            pass
        return queue_lengths

class DQNAgent:
    def __init__(self, n_observations, n_actions, batch_size=128, gamma=0.99, 
                 eps_start=1.0, eps_end=0.01, eps_decay=0.997, target_update=5, 
                 memory_size=50000, learning_rate=0.0001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.memory = SimpleReplayMemory(memory_size)  # Use the simplified memory
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.n_actions = n_actions
        self.steps_done = 0

    def get_action(self, state):
        sample = random.random()
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        with torch.no_grad():
            if sample > self.eps_threshold:
                self.policy_net.eval()  # Set to evaluation mode
                action = self.policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
                self.policy_net.train()  # Set back to training mode
                return action
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], 
                                  device=self.device, dtype=torch.long)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)  # Sample from the simplified memory
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                              batch.next_state)), 
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state 
                                         if s is not None]).to(self.device)
        
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat([torch.tensor([r], device=self.device) 
                                for r in batch.reward])

        # Compute current Q values
        self.policy_net.train()  # Ensure training mode
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            self.target_net.eval()  # Ensure eval mode
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['state_dict'])
            else:
                print("Warning: Unexpected checkpoint structure. Attempting to load directly.")
                self.policy_net.load_state_dict(checkpoint)
        else:
            print("Warning: Checkpoint is not a dictionary. Attempting to load directly.")
            self.policy_net.load_state_dict(checkpoint)
        
        self.policy_net = self.policy_net.to('cpu')
        self.policy_net.eval()

def train(episodes=10):
    # Initialize environment
    env = TrafficEnvironment(os.path.join(os.path.dirname(__file__), "traditional_traffic.sumo.cfg"))
    
    agent = DQNAgent(
        n_observations=42,  # Match the new state dimension
        n_actions=8,
        batch_size=64,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.998,
        target_update=10,
        memory_size=10000,
        learning_rate=0.001
    )

    all_metrics = {
        'rewards': [],
        'waiting_times': [],
        'speeds': [],
        'vehicles': [],
        'queue_lengths': []
    }

    route_file = generate_new_random_traffic()
    start_time = time.time()

    print("\nStarting Training...")
    print("=" * 80)

    for episode in range(episodes):
        state = env.reset(route_file)
        total_reward = 0
        step_count = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action.item())
            
            agent.memory.push(state, action, next_state, torch.tensor([reward], device=device))
            agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                break

        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        waiting_time, avg_speed, avg_vehicles = env.get_performance_metrics()
        
        # Store metrics
        all_metrics['rewards'].append(total_reward)
        all_metrics['waiting_times'].append(waiting_time)
        all_metrics['speeds'].append(avg_speed)
        all_metrics['vehicles'].append(avg_vehicles)
        all_metrics['queue_lengths'].append(env.episode_metrics.get('queue_lengths', []))

        # Print episode summary with step count
        print(f"Episode {episode + 1}/{episodes} | "
              f"Steps: {step_count} | "
              f"Reward: {total_reward:>8.1f} | "
              f"Wait Time: {waiting_time:>6.1f}s | "
              f"Avg Speed: {avg_speed:>4.1f}m/s | "
              f"Vehicles: {avg_vehicles:>3.0f}")

        if (episode + 1) % 10 == 0:
            route_file = generate_new_random_traffic()

    # Print final summary
    elapsed_time = time.time() - start_time
    print("\nTraining Complete!")
    print("=" * 80)
    print(f"Total Training Time: {elapsed_time:.1f} seconds")
    print(f"Average Metrics across {episodes} episodes:")
    print(f"Average Reward: {sum(all_metrics['rewards'])/episodes:>8.1f}")
    print(f"Average Waiting Time: {sum(all_metrics['waiting_times'])/episodes:>6.1f}s")
    print(f"Average Speed: {sum(all_metrics['speeds'])/episodes:>4.1f}m/s")
    print(f"Average Vehicles: {sum(all_metrics['vehicles'])/episodes:>3.0f}")
    print("=" * 80)

    # Save the trained agent
    model_filename = f'trained_dqn_agent_episodes_{episodes}.pth'
    agent.save(model_filename)
    
    return agent, all_metrics['rewards'], all_metrics['waiting_times'], all_metrics['speeds']

def plot_training_results(episode_rewards, episode_waiting_times, episode_avg_speeds):
    episodes = range(1, len(episode_rewards) + 1)

    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(131)
    plt.plot(episodes, episode_rewards, 'b-')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    # Plot waiting times
    plt.subplot(132)
    plt.plot(episodes, episode_waiting_times, 'r-')
    plt.title('Average Waiting Times')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.grid(True)

    # Plot speeds
    plt.subplot(133)
    plt.plot(episodes, episode_avg_speeds, 'g-')
    plt.title('Average Speeds')
    plt.xlabel('Episode')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)

    plt.tight_layout()
    
    # Save with higher DPI for better quality
    plt.savefig('training_result.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_saved_model(model_path, num_episodes=5):
    """
    Test a saved DQN model on the traffic environment
    
    Args:
        model_path (str): Path to the saved model file (.pth)
        num_episodes (int): Number of episodes to test
    """
    # Initialize environment
    env = TrafficEnvironment(os.path.join(os.path.dirname(__file__), "traditional_traffic.sumo.cfg"))
    
    # Force CPU device for testing
    device = torch.device('cpu')
    
    # Initialize agent with same parameters
    agent = DQNAgent(
        n_observations=42,
        n_actions=8,
        batch_size=64,
        gamma=0.95,
        eps_start=0,  # No exploration during testing
        eps_end=0,
        eps_decay=1,
        target_update=10,
        memory_size=10000,
        learning_rate=0.001
    )
    
    # Load the saved model and ensure it's on CPU
    agent.load(model_path)
    agent.policy_net = agent.policy_net.to(device)
    agent.policy_net.eval()
    
    metrics = {
        'waiting_times': [],
        'speeds': [],
        'vehicles': []
    }
    
    print(f"\nTesting saved model from: {model_path}")
    print("=" * 80)
    
    for episode in range(num_episodes):
        route_file = generate_new_random_traffic()
        state = env.reset(route_file)
        state = state.to(device)  # Ensure state is on CPU
        episode_reward = 0
        
        while True:
            # Get action from the model (no exploration)
            with torch.no_grad():
                action = agent.policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
            
            # Take action in environment
            next_state, reward, done = env.step(action.item())
            next_state = next_state.to(device)  # Ensure next_state is on CPU
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Get episode metrics
        waiting_time, avg_speed, avg_vehicles = env.get_performance_metrics()
        metrics['waiting_times'].append(waiting_time)
        metrics['speeds'].append(avg_speed)
        metrics['vehicles'].append(avg_vehicles)
        
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:>8.1f} | "
              f"Wait Time: {waiting_time:>6.1f}s | "
              f"Avg Speed: {avg_speed:>4.1f}m/s | "
              f"Vehicles: {avg_vehicles:>3.0f}")
    
    # Print average metrics
    print("\nTest Results:")
    print("=" * 80)
    print(f"Average Waiting Time: {sum(metrics['waiting_times'])/num_episodes:>6.1f}s")
    print(f"Average Speed: {sum(metrics['speeds'])/num_episodes:>4.1f}m/s")
    print(f"Average Vehicles: {sum(metrics['vehicles'])/num_episodes:>3.0f}")
    
    return metrics

if __name__ == "__main__":
    # episodes = 100
    # trained_agent, rewards, waiting_times, avg_speeds = train(episodes=episodes)
    # plot_training_results(rewards, waiting_times, avg_speeds)
    # print("Training completed. Results plotted in 'training_results.png'")
    # print(f"Trained model saved as 'trained_dqn_agent_episodes_{episodes}.pth'")

    # Path to your saved model
    model_path = 'trained_dqn_agent_episodes_100.pth'
    
    # Test the saved model
    test_metrics = test_saved_model(model_path, num_episodes=5)


