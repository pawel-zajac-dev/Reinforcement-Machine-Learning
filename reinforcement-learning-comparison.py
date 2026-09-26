import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. ŚRODOWISKA (ENVIRONMENTS)
# ==========================================
class BernoulliEnvironment:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        # Stały seed dla powtarzalności wyników
        np.random.seed(42) 
        # Losujemy 10 prawdopodobieństw sukcesu z zakresu [0.1, 0.9]
        self.probabilities = np.random.uniform(0.1, 0.9, n_arms)
        self.optimal_arm = np.argmax(self.probabilities)
        self.optimal_reward = np.max(self.probabilities)

    def pull(self, arm):
        # Zwraca 1 (sukces) z prawdopodobieństwem p, inaczej 0
        return 1 if np.random.random() < self.probabilities[arm] else 0

class GaussianEnvironment:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        np.random.seed(42)
        # Losujemy prawdziwe średnie zyski ramion z rozkładu normalnego N(0, 1)
        self.means = np.random.normal(0, 1, n_arms)
        self.optimal_arm = np.argmax(self.means)
        self.optimal_reward = np.max(self.means)

    def pull(self, arm):
        # Zwraca nagrodę z rozkładu N(średnia_ramienia, wariancja=1)
        return np.random.normal(self.means[arm], 1.0)

# ==========================================
# 2. AGENT BAZOWY I EPSILON-GREEDY
# ==========================================
class BaseAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms) # N_t(a)
        self.values = np.zeros(n_arms) # Q_t(a)

    def update(self, action, reward):
        # Aktualizacja średniej kroczącej (Twój autorski wzór)
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = ((n - 1) / n) * value + (1 / n) * reward

class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms) # Eksploracja
        else:
            return np.argmax(self.values)         # Eksploatacja
        
class RandomAgent(BaseAgent):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def select_action(self):
        # Czysta, ślepa eksploracja
        return np.random.randint(self.n_arms) 

class GreedyAgent(BaseAgent):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def select_action(self):
        # Czysta eksploatacja
        # Drobne zabezpieczenie: jeśli agent jeszcze niczego nie wybrał (wszystkie Q=0), losuje
        if np.sum(self.counts) == 0:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)

class EpsilonDecayAgent(BaseAgent):
    def __init__(self, n_arms, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.999):
        super().__init__(n_arms)
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_action(self):
        # 1. Wybór akcji na podstawie obecnego epsilona
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_arms)
        else:
            action = np.argmax(self.values)
            
        # 2. Zanikanie (decay) parametru po każdym kroku
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        return action


class OptimisticGreedyAgent(BaseAgent):
    def __init__(self, n_arms, initial_value=5.0, alpha=0.1):
        super().__init__(n_arms)
        self.values = np.full(n_arms, float(initial_value))
        self.alpha = alpha # Stały krok uczenia

    def select_action(self):
        return np.argmax(self.values)
        
    def update(self, action, reward):
        self.counts[action] += 1
        # Zamiast czystej średniej, używamy stałego kroku alpha
        # Optymizm zderza się z rzeczywistością i wygasa stopniowo
        self.values[action] = self.values[action] + self.alpha * (reward - self.values[action])


class UCB1Agent(BaseAgent):
    def __init__(self, n_arms, c=0.5):
        super().__init__(n_arms)
        self.c = c
        self.t = 0 # Wewnętrzny zegar agenta do liczenia ln(t)

    def select_action(self):
        self.t += 1
        # Krok 1: Wymuszona eksploracja - upewniamy się, że nie dzielimy przez zero
        if 0 in self.counts:
            return np.argmin(self.counts)
        
        # Krok 2: Właściwy wzór UCB1 z marginesem niepewności
        exploration_term = self.c * np.sqrt(np.log(self.t) / self.counts)
        ucb_values = self.values + exploration_term
        
        return np.argmax(ucb_values)


class ThompsonBetaAgent(BaseAgent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Inicjalizacja płaskiego rozkładu a priori (alpha=1, beta=1)
        self.alphas = np.ones(n_arms)
        self.betas = np.ones(n_arms)

    def select_action(self):
        # Próbkowanie wartości z rozkładu Beta dla każdej maszyny
        samples = np.random.beta(self.alphas, self.betas)
        return np.argmax(samples)

    def update(self, action, reward):
        # Aktualizacja parametrów rozkładu sprzężonego (tylko dla środowisk binarnych)
        self.alphas[action] += reward
        self.betas[action] += (1 - reward)
        # Wywołanie bazowego update'u, żeby zachować logowanie zwykłych średnich
        super().update(action, reward)


class ThompsonGaussianAgent(BaseAgent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.precisions = np.ones(n_arms) # Precyzja to odwrotność wariancji (1 / sigma^2)

    def select_action(self):
        # Próbkowanie z rozkładu normalnego na podstawie aktualnej wiedzy
        variances = 1.0 / self.precisions
        samples = np.random.normal(self.means, np.sqrt(variances))
        return np.argmax(samples)

    def update(self, action, reward):
        # Bayesowska aktualizacja a posteriori dla rozkładu normalnego
        old_precision = self.precisions[action]
        old_mean = self.means[action]
        
        new_precision = old_precision + 1.0
        new_mean = (old_precision * old_mean + reward) / new_precision
        
        self.precisions[action] = new_precision
        self.means[action] = new_mean
        
        super().update(action, reward)
    
# ==========================================
# 3. PĘTLA UŚREDNIAJĄCA (MULTIPLE RUNS)
# ==========================================
def run_experiment(env, agent, n_trials=10000):
    rewards = np.zeros(n_trials)
    optimal_actions = np.zeros(n_trials)
    regrets = np.zeros(n_trials)

    for t in range(n_trials):
        action = agent.select_action()
        reward = env.pull(action)
        agent.update(action, reward)

        rewards[t] = reward
        optimal_actions[t] = 1 if action == env.optimal_arm else 0
        
        if hasattr(env, 'probabilities'):
            regrets[t] = env.optimal_reward - env.probabilities[action]
        else:
            regrets[t] = env.optimal_reward - env.means[action]

    return rewards, optimal_actions, regrets

def run_multiple_experiments(env_class, agent_class, n_runs=100, n_trials=10000, **agent_kwargs):
    all_rewards = np.zeros((n_runs, n_trials))
    all_optimals = np.zeros((n_runs, n_trials))
    all_regrets = np.zeros((n_runs, n_trials))

    for r in range(n_runs):
        # Unikalne ziarno dla każdego przebiegu (nowe losowanie parametrów maszyn)
        np.random.seed(42 + r) 
        env = env_class(n_arms=10)
        
        # Resetujemy wewnętrzne ziarno agenta dla czystości testu
        np.random.seed(1000 + r)
        agent = agent_class(n_arms=10, **agent_kwargs)

        rewards, optimals, regrets = run_experiment(env, agent, n_trials)
        all_rewards[r] = rewards
        all_optimals[r] = optimals
        all_regrets[r] = regrets

    # Uśrednienie wyników w pionie (po wszystkich powtórzeniach eksperymentu)
    return all_rewards.mean(axis=0), all_optimals.mean(axis=0), all_regrets.mean(axis=0)

# ==========================================
# 4. GENEROWANIE WYKRESÓW PUBLIKACYJNYCH
# ==========================================
def plot_academic_results(results_dict, env_name):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for name, (rewards, optimals, regrets) in results_dict.items():
        axs[0].plot(rewards, label=name, alpha=0.8)
        axs[1].plot(optimals * 100, label=name, alpha=0.8)
        axs[2].plot(np.cumsum(regrets), label=name, linewidth=2)

    axs[0].set_title(f"Średnia nagroda ({env_name})")
    axs[0].set_ylabel("Nagroda")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    axs[1].set_title(f"% Optymalnych Wyborów ({env_name})")
    axs[1].set_ylabel("Procent [%]")
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    axs[2].set_title(f"Skumulowany Żal ({env_name})")
    axs[2].set_ylabel("Żal (Regret)")
    axs[2].set_xlabel("Krok czasowy (Trials)")
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. GŁÓWNY BLOK WYKONAWCZY
# ==========================================
if __name__ == "__main__":
    n_runs = 100    
    n_trials = 10000
    
    print(f"Rozpoczynam uśrednianie {n_runs} eksperymentów po {n_trials} prób...")

    # ==========================================
    # ŚRODOWISKO BERNOULLIEGO
    # ==========================================
    print("\n=== ŚRODOWISKO BERNOULLIEGO ===")
    naive_agents_bern = {
        "Random": (RandomAgent, {}),
        "Greedy (0)": (GreedyAgent, {}),
        "Epsilon-Greedy (0.1)": (EpsilonGreedyAgent, {"epsilon": 0.1}),
        "Epsilon-Decay": (EpsilonDecayAgent, {}),
        "Optimistic Greedy": (OptimisticGreedyAgent, {})
    }
    advanced_agents_bern = {
        "Epsilon-Greedy (0.1)": (EpsilonGreedyAgent, {"epsilon": 0.1}),
        "UCB1 (c=0.5)": (UCB1Agent, {"c": 0.5}),
        "Thompson Beta": (ThompsonBetaAgent, {})
    }

    naive_results_bern = {}
    for name, (agent_class, kwargs) in naive_agents_bern.items():
        print(f"Trenowanie Naiwne (Bernoulli): {name}...")
        naive_results_bern[name] = run_multiple_experiments(BernoulliEnvironment, agent_class, n_runs=n_runs, n_trials=n_trials, **kwargs)
        
    advanced_results_bern = {}
    for name, (agent_class, kwargs) in advanced_agents_bern.items():
        print(f"Trenowanie Zaawansowane (Bernoulli): {name}...")
        advanced_results_bern[name] = run_multiple_experiments(BernoulliEnvironment, agent_class, n_runs=n_runs, n_trials=n_trials, **kwargs)

    # ==========================================
    # ŚRODOWISKO GAUSSA
    # ==========================================
    print("\n=== ŚRODOWISKO GAUSSA ===")
    naive_agents_gauss = {
        "Random": (RandomAgent, {}),
        "Greedy (0)": (GreedyAgent, {}),
        "Epsilon-Greedy (0.1)": (EpsilonGreedyAgent, {"epsilon": 0.1}),
        "Epsilon-Decay": (EpsilonDecayAgent, {}),
        "Optimistic Greedy": (OptimisticGreedyAgent, {})
    }
    advanced_agents_gauss = {
        "Epsilon-Greedy (0.1)": (EpsilonGreedyAgent, {"epsilon": 0.1}),
        "UCB1 (c=0.5)": (UCB1Agent, {"c": 0.5}),
        "Thompson Gaussian": (ThompsonGaussianAgent, {}) # Zmiana na wersję ciągłą!
    }

    naive_results_gauss = {}
    for name, (agent_class, kwargs) in naive_agents_gauss.items():
        print(f"Trenowanie Naiwne (Gauss): {name}...")
        naive_results_gauss[name] = run_multiple_experiments(GaussianEnvironment, agent_class, n_runs=n_runs, n_trials=n_trials, **kwargs)
        
    advanced_results_gauss = {}
    for name, (agent_class, kwargs) in advanced_agents_gauss.items():
        print(f"Trenowanie Zaawansowane (Gauss): {name}...")
        advanced_results_gauss[name] = run_multiple_experiments(GaussianEnvironment, agent_class, n_runs=n_runs, n_trials=n_trials, **kwargs)

    # Generowanie wykresów
    print("\nGenerowanie wykresów...")
    plot_academic_results(naive_results_bern, "Bernoulli - Metody Naiwne")
    plot_academic_results(advanced_results_bern, "Bernoulli - Metody Zaawansowane")
    plot_academic_results(naive_results_gauss, "Gauss - Metody Naiwne")
    plot_academic_results(advanced_results_gauss, "Gauss - Metody Zaawansowane")

    # ==========================================
    # WYŚWIETLANIE WYNIKÓW KOŃCOWYCH DO TABELI
    # ==========================================
    print("\n=== KOŃCOWE WYNIKI (T=10000) ===")
    
    all_bern = {**naive_results_bern, **advanced_results_bern}
    all_gauss = {**naive_results_gauss, **advanced_results_gauss}
    
    print("--- Środowisko Bernoulliego ---")
    for name, metrics in all_bern.items():
        avg_reward = np.mean(metrics[0])          # Uśredniona nagroda z całej symulacji
        final_opt_pct = metrics[1][-1] * 100      
        final_regret = np.sum(metrics[2])         
        print(f"{name}: Śr. Nagroda = {avg_reward:.4f}, Żal = {final_regret:.2f}, Optymalne = {final_opt_pct:.2f}%")

    print("\n--- Środowisko Gaussa ---")
    for name, metrics in all_gauss.items():
        avg_reward = np.mean(metrics[0])          # Uśredniona nagroda z całej symulacji
        final_opt_pct = metrics[1][-1] * 100      
        final_regret = np.sum(metrics[2])         
        print(f"{name}: Śr. Nagroda = {avg_reward:.4f}, Żal = {final_regret:.2f}, Optymalne = {final_opt_pct:.2f}%")
