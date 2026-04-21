import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

class SignalingGovernanceEnv(AECEnv):
    """
    Environnement de type 'Signaling Game' pour tester la Gouvernance IA.
    Agents:
    - 'requerant_0' : Connaît son profil (Honnête/Fraudeur), émet un signal.
    - 'orchestrateur_0' : Observe le signal, décide d'accepter ou bloquer.
    """
    metadata = {"render_modes": ["human"], "name": "signaling_governance_v1"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Définition des agents
        self.agents = ["requerant_0", "orchestrateur_0"]
        self.possible_agents = self.agents[:]
        
        # Espaces d'actions
        # Requerant : 0 (Emettre Signal Clair), 1 (Emettre Signal Ambigu)
        # Orchestrateur : 0 (Accepter), 1 (Bloquer)
        self.action_spaces = {agent: Discrete(2) for agent in self.agents}
        
        # Espaces d'observations
        # Requerant voit son propre type caché : 0 (Honnête), 1 (Fraudeur)
        self.observation_spaces = {
            "requerant_0": Discrete(2),
            # Orchestrateur voit le signal du requérant : 0 (Clair), 1 (Ambigu), ou 2 (En attente)
            "orchestrateur_0": Discrete(3) 
        }

        # Probabilité qu'un requérant soit honnête (ex: 80% de la population)
        self.prob_honest = 0.8
        
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # L'environnement tire au sort le profil caché du requérant
        if seed is not None:
            np.random.seed(seed)
        self.hidden_type = 0 if np.random.rand() < self.prob_honest else 1
        
        # Initialisation de l'observation (2 = pas encore de signal émis)
        self.current_signal = 2 
        
        # Le requérant joue toujours en premier
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        if agent == "requerant_0":
            # Le requérant observe son propre type caché
            return np.array(self.hidden_type, dtype=np.int64)
        elif agent == "orchestrateur_0":
            # L'orchestrateur observe uniquement le signal émis par le requérant
            return np.array(self.current_signal, dtype=np.int64)

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        if agent == "requerant_0":
            # Le requérant choisit son signal (0: Clair, 1: Ambigu)
            # Dans un jeu complet, il y a des contraintes. 
            # Ex: Un fraudeur a du mal à émettre un signal parfait sans erreur.
            self.current_signal = action
            
            # On passe le tour à l'orchestrateur
            self.agent_selection = self._agent_selector.next()

        elif agent == "orchestrateur_0":
            # L'orchestrateur décide de son action face au signal (0: Accepter, 1: Bloquer)
            orchestrator_action = action
            
            # --- MATRICE DES RÉCOMPENSES (La Gouvernance) ---
            if self.hidden_type == 0:  # Le requérant était HONNÊTE
                if orchestrator_action == 0: # Accepter
                    self.rewards["orchestrateur_0"] = 1
                    self.rewards["requerant_0"] = 1
                else: # Bloquer (Erreur de gouvernance : Faux Positif)
                    self.rewards["orchestrateur_0"] = -2
                    self.rewards["requerant_0"] = -1
                    
            elif self.hidden_type == 1: # Le requérant était FRAUDEUR
                if orchestrator_action == 0: # Accepter (Faille de sécurité : Faux Négatif)
                    self.rewards["orchestrateur_0"] = -3
                    self.rewards["requerant_0"] = 3
                else: # Bloquer (Bonne régulation : Vrai Positif)
                    self.rewards["orchestrateur_0"] = 2
                    self.rewards["requerant_0"] = -2

            # Fin du jeu après la décision de l'orchestrateur
            self.terminations = {a: True for a in self.agents}
            self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()

# Test rapide pour vérifier que l'environnement fonctionne
if __name__ == "__main__":
    from pettingzoo.test import api_test
    env = SignalingGovernanceEnv()
    api_test(env, num_cycles=1000, verbose_progress=False)
    print("✅ L'environnement PettingZoo est valide et prêt pour l'IA !")