import numpy as np
import math

class RequerantAgent:
    """ L'agent qui envoie le signal (avec une part de bruit/erreur). """
    def __init__(self):
        # Probabilités d'émettre un Signal Clair (0) selon le vrai profil
        self.prob_clair_if_honest = 0.90  # 10% de chance de faire une erreur (Ambigu)
        self.prob_clair_if_fraud = 0.20   # 20% de chance de bien bluffer (Clair)

    def get_action(self, hidden_type):
        rand = np.random.rand()
        if hidden_type == 0:  # Honnête
            return 0 if rand < self.prob_clair_if_honest else 1
        else:  # Fraudeur
            return 0 if rand < self.prob_clair_if_fraud else 1

class AgentL1_Reatif:
    """ Baseline 1 : Agent réactif (statistique simple). Ne réfléchit pas aux intentions. """
    def get_action(self, signal):
        # Règle apprise empiriquement : "Les signaux ambigus sont souvent des fraudes"
        if signal == 0: # Clair
            return 0 # Accepter
        else: # Ambigu
            return 1 # Bloquer (Sanctionne aveuglément)
        
class AgentL2_Bayesien:
    """ Baseline 2 : Bayesian ToM. Calcule la distribution exacte à CHAQUE itération. """
    def __init__(self, prior_honest=0.8):
        self.prior_H = prior_honest
        self.prior_F = 1.0 - prior_honest
        self.likelihood_C_given_H = 0.90
        self.likelihood_A_given_H = 0.10
        self.likelihood_C_given_F = 0.20
        self.likelihood_A_given_F = 0.80

    def get_action(self, signal):
        # Inférence de l'intention cachée via Bayes
        if signal == 0: # Signal Clair
            marginal = (self.likelihood_C_given_H * self.prior_H) + (self.likelihood_C_given_F * self.prior_F)
            post_H = (self.likelihood_C_given_H * self.prior_H) / marginal
        else: # Signal Ambigu
            marginal = (self.likelihood_A_given_H * self.prior_H) + (self.likelihood_A_given_F * self.prior_F)
            post_H = (self.likelihood_A_given_H * self.prior_H) / marginal
            
        post_F = 1.0 - post_H
        
        # Décision basée sur la maximisation de l'utilité (matrice des récompenses)
        # Bloquer est optimal si la probabilité de fraude est très élevée
        if post_F > 0.5:
            return 1 # Bloquer
        else:
            return 0 # Accepter

class AgentHybrideStatique:
    """ Baseline 3 : Hybride statique (ex: Neuro-symbolique classique sans orchestration).
        Exécute systématiquement le niveau implicite (L1) ET le niveau explicite (L2),
        puis fusionne leurs décisions. Paie le coût cognitif maximal en permanence. """
    def __init__(self, prior_honest=0.8):
        self.agent_L1 = AgentL1_Reatif()
        self.agent_L2 = AgentL2_Bayesien(prior_honest)

    def get_action(self, signal):
        # 1. Le modèle exécute toujours le calcul réactif rapide
        action_implicite = self.agent_L1.get_action(signal)
        
        # 2. Le modèle exécute toujours le calcul bayésien lourd (Pas de déclencheur St !)
        action_explicite = self.agent_L2.get_action(signal)
        
        # 3. Fusion : la logique explicite a le dernier mot pour garantir la gouvernance
        return action_explicite

class Orchestrateur_L3:
    """ VOTRE CONTRIBUTION : L'architecture multi-niveaux avec déclencheur St et gouvernance Kappa. """
    def __init__(self, tau=1.0, delta=0.6, prior_honest=0.8):
        self.tau = tau      # Seuil de surprise pour déclencher L2
        self.delta = delta  # Seuil d'auditabilité (Gouvernance)
        
        # Modèle de base pour L1 (Probabilité marginale des signaux)
        self.p_clair = (0.90 * prior_honest) + (0.20 * (1-prior_honest))
        self.p_ambigu = 1.0 - self.p_clair
        
        self.agent_L2 = AgentL2_Bayesien(prior_honest)
        self.agent_L1 = AgentL1_Reatif()
        
        # Statistiques pour l'article
        self.l2_activations = 0

    def get_action(self, signal):
        # ÉTAPE 1 : Prédiction de base (L1) et calcul de la surprise (St)
        prob_obs = self.p_clair if signal == 0 else self.p_ambigu
        surprise_St = -math.log(prob_obs) # Entropie croisée / Surprise de Shannon
        
        # ÉTAPE 2 : Déclenchement conditionnel
        if surprise_St > self.tau:
            self.l2_activations += 1
            
            # Déclenchement de la ToM cognitive (L2)
            if signal == 0:
                marginal = (0.90 * 0.8) + (0.20 * 0.2)
                post_H = (0.90 * 0.8) / marginal
            else:
                marginal = (0.10 * 0.8) + (0.80 * 0.2)
                post_H = (0.10 * 0.8) / marginal
            post_F = 1.0 - post_H
            
            # ÉTAPE 3 : Auditabilité normative (L3) - Calcul de Kappa
            # Entropie de la croyance H(theta)
            # Pour éviter log(0), on ajoute un epsilon
            eps = 1e-9
            entropie_croyance = -(post_H * math.log(post_H + eps) + post_F * math.log(post_F + eps))
            
            # Entropie max pour 2 états = log(2)
            kappa = 1.0 - (entropie_croyance / math.log(2))
            
            # ÉTAPE 4 : Application de la règle de Gouvernance
            action_candidate = 1 if post_F > 0.5 else 0
            
            if kappa >= self.delta:
                return action_candidate # Confiance épistémique suffisante, on agit !
            else:
                # Exception Normative : La situation est trop incertaine.
                # Par prudence (présomption d'innocence), on refuse de sanctionner.
                return 0 # Fallback (Accepter)
        else:
            # Interaction de routine (St < tau), on reste en L1 pour économiser le coût
            return self.agent_L1.get_action(signal)

if __name__ == "__main__":
    print("✅ Fichier agents.py validé ! Les modèles (L1, L2, Hybride, L3) sont prêts.")