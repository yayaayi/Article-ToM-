import numpy as np
import matplotlib.pyplot as plt
from env_signaling import SignalingGovernanceEnv
from agents import RequerantAgent, AgentL1_Reatif, AgentL2_Bayesien, AgentHybrideStatique, Orchestrateur_L3

def run_simulation(agent_name, num_episodes=5000):
    env = SignalingGovernanceEnv()
    requerant = RequerantAgent()
    
    # Instanciation de l'agent testé
    if agent_name == "L1":
        orchestrateur = AgentL1_Reatif()
    elif agent_name == "L2":
        orchestrateur = AgentL2_Bayesien()
    elif agent_name == "Hybride":
        orchestrateur = AgentHybrideStatique() 
    elif agent_name == "L3":
        orchestrateur = Orchestrateur_L3(tau=0.25, delta=0.6)
        
    historique_recompenses = []
    historique_couts = []
    
    recompense_cumulee = 0
    cout_cumule = 0
    
    for episode in range(num_episodes):
        env.reset()
        
        while not all(env.terminations.values()):
            agent = env.agent_selection
            obs = env.observe(agent)
            
            if agent == "requerant_0":
                action = requerant.get_action(obs)
            elif agent == "orchestrateur_0":
                action = orchestrateur.get_action(obs)
                
                # --- CALCUL DU COÛT COMPUTATIONNEL ---
                if agent_name == "L1":
                    cout_cumule += 1           # Coût L1 seul
                elif agent_name == "L2":
                    cout_cumule += 10          # Coût L2 seul
                elif agent_name == "Hybride":
                    cout_cumule += 11          # Coût L1 + L2 systématique
                elif agent_name == "L3":
                    # L3 coûte 1 en routine, mais 11 s'il déclenche L2 (anomalie)
                    if obs == 1: 
                        cout_cumule += 11
                    else:
                        cout_cumule += 1
            
            env.step(action)
            
        # Enregistrement à la fin de chaque épisode
        recompense_cumulee += env.rewards["orchestrateur_0"]
        historique_recompenses.append(recompense_cumulee)
        historique_couts.append(cout_cumule)
        
    return historique_recompenses, historique_couts

if __name__ == "__main__":
    print("🚀 Lancement de la simulation avec 4 modèles (5000 épisodes)...")
    
    episodes = 5000
    rew_L1, cost_L1 = run_simulation("L1", episodes)
    rew_L2, cost_L2 = run_simulation("L2", episodes)
    rew_Hyb, cost_Hyb = run_simulation("Hybride", episodes)
    rew_L3, cost_L3 = run_simulation("L3", episodes)
    
    print("✅ Simulation terminée ! Génération des graphiques...")

    # --- GÉNÉRATION DES GRAPHIQUES POUR L'ARTICLE ---
    # Utilisation d'un style propre et académique
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique 1 : Performance (Récompenses)
    ax1.plot(rew_L1, label="Baseline 1: MARL ($L_1$)", color='red', linestyle='--', alpha=0.8)
    ax1.plot(rew_L2, label="Baseline 2: BToM ($L_2$)", color='blue', linestyle='-.', alpha=0.8)
    ax1.plot(rew_Hyb, label="Baseline 3: Hybride Statique", color='purple', linestyle=':', linewidth=2)
    ax1.plot(rew_L3, label="Proposé: Orchestrateur ($L_0-L_3$)", color='green', linewidth=2.5)
    ax1.set_title("Performance Stratégique et Normative", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Nombre d'épisodes", fontsize=12)
    ax1.set_ylabel("Utilité cumulative (Institution)", fontsize=12)
    ax1.legend(loc="upper left")

    # Graphique 2 : Coût Computationnel
    ax2.plot(cost_L1, label="Baseline 1: MARL ($L_1$)", color='red', linestyle='--', alpha=0.8)
    ax2.plot(cost_L2, label="Baseline 2: BToM ($L_2$)", color='blue', linestyle='-.', alpha=0.8)
    ax2.plot(cost_Hyb, label="Baseline 3: Hybride Statique", color='purple', linestyle=':', linewidth=2)
    ax2.plot(cost_L3, label="Proposé: Orchestrateur ($L_0-L_3$)", color='green', linewidth=2.5)
    ax2.set_title("Coût Computationnel Cumulé", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Nombre d'épisodes", fontsize=12)
    ax2.set_ylabel("Unités d'inférence ($\sum \lambda C$)", fontsize=12)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    
    # Sauvegarde de l'image en haute résolution pour Word/LaTeX
    plt.savefig("resultats_gouvernance.png", dpi=300)
    print("📊 Graphique sauvegardé sous 'resultats_gouvernance.png'")
    
    # Affichage de la fenêtre interactive
    plt.show()