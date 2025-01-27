import numpy as np

class Belief:
    mean: float
    precision: float
    
    def __init__(self, mean: float = None, precision: float = 0) -> None:
        self.mean = mean
        self.precision = precision
    
    def __add__(self, other: 'Belief') -> 'Belief':
        if self.precision == 0:
            return other
        if other.precision == 0:
            return self
        
        new_precision = self.precision + other.precision
        new_mean = (self.mean * self.precision + other.mean * other.precision) / new_precision
        return Belief(new_mean, new_precision)
    
    def __mul__(self, coef: float) -> 'Belief':
        return Belief(self.mean, self.precision * coef)
    

class Agent:
    private_belief: Belief  # Only counting its private information
    public_belief: Belief  # All-things-considered belief, including updates from other agents' reports
    
    def __init__(self, initial_belief: Belief) -> None:
        self.private_belief = initial_belief
        self.public_belief = initial_belief
    
    def observe_private_signal(self, signal: float, precision: float) -> None:
        self.private_belief = self.private_belief + Belief(signal, precision)
    
    def observe_partner_report(self, observed_belief: Belief) -> None:
        self.public_belief = self.private_belief + observed_belief


if __name__ == '__main__':
    num_agents = 10
    num_rounds = 1000
    true_answer = 42
    lambda_1 = 0.11
    lambda_2 = 1
    
    def generate_signal(precision: float) -> float:
        return np.random.normal(true_answer, 1 / precision)
    
    agents = [Agent(Belief(generate_signal(1), 1)) for _ in range(num_agents)]
    
    for round_id in range(num_rounds):
        lm_belief = sum((agent.public_belief for agent in agents), start=Belief()) * lambda_1
        print(f"Round {round_id}: {lm_belief.mean} {lm_belief.precision}")
        
        for agent in agents:
            agent.observe_partner_report(lm_belief * lambda_2)
            agent.observe_private_signal(generate_signal(1), 1)
        