import numpy as np
import math
from decimal import Decimal
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

np.random.seed(1900)

class Belief:
    mean: Decimal
    precision: Decimal
    
    def __init__(self, mean: Decimal | float = 12345, precision: Decimal | float = 0) -> None:
        self.mean = Decimal(mean)
        self.precision = Decimal(precision)
    
    def __add__(self, other: 'Belief') -> 'Belief':
        new_precision = self.precision + other.precision
        new_mean = (self.mean * self.precision + other.mean * other.precision) / new_precision
        return Belief(new_mean, new_precision)
    
    def __mul__(self, coef: Decimal | float) -> 'Belief':
        return Belief(self.mean, self.precision * coef)
    

class Agent:
    private_belief: Belief  # Only counting its private information
    public_belief: Belief  # All-things-considered belief, including updates from other agents' reports
    
    def __init__(self, initial_belief: Belief) -> None:
        self.private_belief = initial_belief
        self.public_belief = initial_belief
    
    def observe_private_signal(self, signal: Decimal | float, precision: Decimal | float) -> None:
        self.private_belief = self.private_belief + Belief(signal, precision)
    
    def observe_partner_report(self, observed_belief: Belief) -> None:
        self.public_belief = self.private_belief + observed_belief

num_agents = 10
num_rounds = 500
true_answer = .42
lambda_1 = Decimal("0.11")
lambda_2 = Decimal("1")
private_precision = Decimal("1")

def generate_signal(precision: Decimal | float) -> float:
    return np.random.normal(true_answer, float(precision) ** -0.5)

def simulate_trajectory() -> List[Belief]:
    agents = [Agent(Belief(generate_signal(private_precision), private_precision)) for _ in range(num_agents)]
    res = []
    
    for round_id in range(num_rounds):
        lm_belief = sum((agent.public_belief for agent in agents), start=Belief()) * lambda_1
        print(f"Round {round_id}: {lm_belief.mean} {math.log(lm_belief.precision):.2f}")
        res.append(lm_belief)
        
        for agent in agents:
            agent.observe_partner_report(lm_belief * lambda_2)
            agent.observe_private_signal(generate_signal(private_precision), private_precision)
    
    return res

def visualize() -> None:
    # Set R-style aesthetics
    # plt.style.use('seaborn')
    # sns.set_palette("husl")
    plt.style.use('ggplot')

    # Generate simulated data
    n_simulations = 15

    # Generate posterior mean trajectories (sigmoid-like curves with noise)
    posterior_means = []
    posterior_entropies = []
    posterior_sigmas = []
    
    for _ in range(n_simulations):
        belief_seq = simulate_trajectory()
        posterior_means.append([float(belief.mean) for belief in belief_seq])
        posterior_entropies.append([(math.log(2 * math.pi * math.e) - math.log(belief.precision)) / 2 for belief in belief_seq])
        posterior_sigmas.append([float(belief.precision ** Decimal(-0.5)) for belief in belief_seq])
    
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 7.5))
    ax1.set_xlim((-2, len(posterior_means[0])-1))

    # Plot posterior mean trajectories
    for pm, ps in zip(posterior_means, posterior_sigmas):
        ax1.plot(pm, color='red', alpha=0.3, linewidth=2)
        ax1.fill_between(list(range(len(pm))), np.array(pm) - np.array(ps) * 2, np.array(pm) + np.array(ps) * 2, alpha=0.25)

    # Format primary axis
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Posterior Mean', fontsize=12)
    ax1.set_ylim((0.22, 0.62))
    ax1.set_yticks(np.arange(0.22, 0.62, 0.05))
    ax1.hlines(y = true_answer, xmin = 0, xmax = len(posterior_means[0]), linestyles='-.', alpha=0.6, color='darkblue', linewidth=3, label='Truth Answer')
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create secondary axis for entropy
    ax2 = ax1.twinx()
    ax2.plot(posterior_entropies[0], color='darkred', linewidth=2.5, linestyle='--', label='Average Posterior Entropy')
    ax2.set_ylabel('Posterior Entropy', fontsize=12)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(False)
    ax2.set_xlim((-2, len(posterior_means[0])-1))

    # Add highlight and annotation
    # ax2.annotate('Diversity Loss', xy=(0.2, 0.8), xytext=(0.23, 0.87), xycoords="figure fraction",
                # arrowprops=dict(facecolor='darkred', shrink=0.05),
                # fontsize=13, color='darkred')

    # Add legend
    lines = [plt.Line2D([0], [0], color='red', alpha=0.3, linewidth=2),
            plt.Line2D([0], [0], color='darkred', linewidth=2.5, linestyle='--'),
            plt.Line2D([0], [0], color='darkblue', linewidth=3, linestyle='-.')]
    ax1.legend(lines, ['Posterior Mean Trajectories (with 95% CI)', 'Average Posterior Entropy', 'Ground Truth'], 
            loc='upper right', fontsize=10)

    # Add title and adjust layout
    plt.title('Bayesian Updating Dynamics: Posterior Mean and Entropy', fontsize=14, pad=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./fig{float(num_agents * lambda_1 * lambda_2):.1f}.pdf")

if __name__ == '__main__':
    visualize()
        