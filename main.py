import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# REPRODUCTIBILITÉ STRICTE (Fixation de la graine)
# ==========================================
np.random.seed(42) # Force Python à donner TOUJOURS les mêmes résultats

# ==========================================
# HYPERPARAMÈTRES DE LA SIMULATION
# ==========================================
NOMBRE_EPISODES = 5000
COUT_L1 = 1.0       
COUT_L2 = 10.0      
TAU = 0.25          
DELTA = 0.60        

# Dynamique de population
PROBA_HONNETE = 0.80
PROBA_FRAUDEUR = 0.20
BRUIT_HONNETE = 0.10    
DISSIMULATION = 0.20    

# Utilités institutionnelles
RECOMPENSE_VRAI_POSITIF = 10
PENALITE_FAUX_POSITIF = -50
RECOMPENSE_VRAI_NEGATIF = 1
PENALITE_FAUX_NEGATIF = -20

# ==========================================
# CLASSES : ENVIRONNEMENT ET AGENTS
# ==========================================

class SignalingGame:
    def generer_interaction(self):
        est_fraudeur = np.random.rand() < PROBA_FRAUDEUR
        if not est_fraudeur:
            signal_ambigu = np.random.rand() < BRUIT_HONNETE
        else:
            signal_ambigu = np.random.rand() < DISSIMULATION
        return est_fraudeur, signal_ambigu

class ArchitecturesToM:
    
    @staticmethod
    def baseline_1_reactive(signal_ambigu):
        cout = COUT_L1
        action = True if signal_ambigu else False
        return action, cout, False, False

    @staticmethod
    def baseline_2_bayesienne(est_fraudeur):
        cout = COUT_L2
        action = True if est_fraudeur else False
        return action, cout, True, False

    @staticmethod
    def baseline_3_hybride(signal_ambigu, est_fraudeur):
        cout = COUT_L1 + COUT_L2
        action = True if est_fraudeur else False
        return action, cout, True, False

    @staticmethod
    def orchestrateur_L0_L3(signal_ambigu, est_fraudeur):
        cout = COUT_L1
        surprise_St = 1.0 if signal_ambigu else 0.0
        
        l2_declenche = False
        l3_veto = False
        
        if surprise_St > TAU:
            l2_declenche = True
            cout += COUT_L2
            
            kappa = 0.85 if est_fraudeur else 0.40
                
            if kappa >= DELTA:
                action = True
            else:
                action = False
                l3_veto = True
        else:
            action = False
            
        return action, cout, l2_declenche, l3_veto

def calculer_utilite(action_sanction, est_fraudeur):
    if action_sanction and est_fraudeur:
        return RECOMPENSE_VRAI_POSITIF, False
    elif action_sanction and not est_fraudeur:
        return PENALITE_FAUX_POSITIF, True
    elif not action_sanction and not est_fraudeur:
        return RECOMPENSE_VRAI_NEGATIF, False
    elif not action_sanction and est_fraudeur:
        return PENALITE_FAUX_NEGATIF, False

# ==========================================
# BOUCLE PRINCIPALE DE SIMULATION
# ==========================================

if __name__ == "__main__":
    env = SignalingGame()
    modeles = ArchitecturesToM()

    resultats = {
        "L1": {"utilite": [], "cout": [], "faux_positifs": 0, "l2_appels": 0, "l3_vetos": 0},
        "L2": {"utilite": [], "cout": [], "faux_positifs": 0, "l2_appels": 0, "l3_vetos": 0},
        "HYB": {"utilite": [], "cout": [], "faux_positifs": 0, "l2_appels": 0, "l3_vetos": 0},
        "ORC": {"utilite": [], "cout": [], "faux_positifs": 0, "l2_appels": 0, "l3_vetos": 0}
    }

    print(f"Lancement de la simulation sur {NOMBRE_EPISODES} épisodes...")

    for _ in range(NOMBRE_EPISODES):
        est_fraudeur, signal_ambigu = env.generer_interaction()

        # L1
        act, cout, l2_trig, l3_v = modeles.baseline_1_reactive(signal_ambigu)
        u, fp = calculer_utilite(act, est_fraudeur)
        resultats["L1"]["utilite"].append(u - cout); resultats["L1"]["cout"].append(cout)
        if fp: resultats["L1"]["faux_positifs"] += 1
        if l2_trig: resultats["L1"]["l2_appels"] += 1
        if l3_v: resultats["L1"]["l3_vetos"] += 1

        # L2
        act, cout, l2_trig, l3_v = modeles.baseline_2_bayesienne(est_fraudeur)
        u, fp = calculer_utilite(act, est_fraudeur)
        resultats["L2"]["utilite"].append(u - cout); resultats["L2"]["cout"].append(cout)
        if fp: resultats["L2"]["faux_positifs"] += 1
        if l2_trig: resultats["L2"]["l2_appels"] += 1
        if l3_v: resultats["L2"]["l3_vetos"] += 1

        # Hybride
        act, cout, l2_trig, l3_v = modeles.baseline_3_hybride(signal_ambigu, est_fraudeur)
        u, fp = calculer_utilite(act, est_fraudeur)
        resultats["HYB"]["utilite"].append(u - cout); resultats["HYB"]["cout"].append(cout)
        if fp: resultats["HYB"]["faux_positifs"] += 1
        if l2_trig: resultats["HYB"]["l2_appels"] += 1
        if l3_v: resultats["HYB"]["l3_vetos"] += 1

        # Orchestrateur
        act, cout, l2_trig, l3_v = modeles.orchestrateur_L0_L3(signal_ambigu, est_fraudeur)
        u, fp = calculer_utilite(act, est_fraudeur)
        resultats["ORC"]["utilite"].append(u - cout); resultats["ORC"]["cout"].append(cout)
        if fp: resultats["ORC"]["faux_positifs"] += 1
        if l2_trig: resultats["ORC"]["l2_appels"] += 1
        if l3_v: resultats["ORC"]["l3_vetos"] += 1

    # ==========================================
    # AFFICHAGE DES RÉSULTATS DANS LA CONSOLE
    # ==========================================
    
    print("\n" + "="*110)
    print(f"{'ARCHITECTURE':<22} | {'UTILITÉ (± σ)':<18} | {'COÛT (± σ)':<16} | {'FAUX POS.':<10} | {'L2 INVOQUÉ':<12} | {'VETO L3 (sur L2)'}")
    print("="*110)

    for cle, nom in [("L1", "Baseline 1 (L1)"), ("L2", "Baseline 2 (L2)"), ("HYB", "Baseline 3 (Hybride)"), ("ORC", "Orchestrateur L0-L3")]:
        u_tot = np.sum(resultats[cle]["utilite"])
        u_std = np.std(resultats[cle]["utilite"]) * np.sqrt(NOMBRE_EPISODES)
        c_moy = np.mean(resultats[cle]["cout"])
        c_std = np.std(resultats[cle]["cout"])
        tx_fp = (resultats[cle]["faux_positifs"] / NOMBRE_EPISODES) * 100
        tx_l2 = (resultats[cle]["l2_appels"] / NOMBRE_EPISODES) * 100
        
        if resultats[cle]["l2_appels"] > 0:
            tx_veto = (resultats[cle]["l3_vetos"] / resultats[cle]["l2_appels"]) * 100
            veto_str = f"{tx_veto:>4.1f} %"
        else:
            veto_str = "N/A"

        print(f"{nom:<22} | {u_tot:>6.0f} (± {u_std:>4.0f})     | {c_moy:>4.2f} (± {c_std:>4.2f})    | {tx_fp:>5.1f} %    | {tx_l2:>5.1f} %      | {veto_str}")
    print("="*110)

    # ==========================================
    # GÉNÉRATION DES GRAPHIQUES
    # ==========================================
    print("\nGénération des figures en cours...")
    couleurs = {'L1': '#d62728', 'L2': '#2ca02c', 'HYB': '#ff7f0e', 'ORC': '#1f77b4'}

    # --- FIGURE 1 ---
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(resultats["L1"]["utilite"]), label="Baseline 1 (L1)", color=couleurs['L1'], linewidth=2)
    plt.plot(np.cumsum(resultats["L2"]["utilite"]), label="Baseline 2 (L2)", color=couleurs['L2'], linewidth=2)
    
    # LA CORRECTION EST ICI : Ajout de la Baseline 3 (en pointillés dashdot)
    plt.plot(np.cumsum(resultats["HYB"]["utilite"]), label="Baseline 3 (Hybride)", color=couleurs['HYB'], linewidth=2, linestyle='-.')
    
    plt.plot(np.cumsum(resultats["ORC"]["utilite"]), label="Orchestrateur (L0-L3)", color=couleurs['ORC'], linewidth=2, linestyle='--')
    
    plt.title("Figure 1 : Évolution de l'Utilité Cumulative", fontsize=14, fontweight='bold')
    plt.xlabel("Épisodes d'interaction")
    plt.ylabel("Utilité institutionnelle cumulative")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show() 

    # --- FIGURE 2 ---
    plt.figure(figsize=(8, 6))
    architectures = ['Baseline 1\n(Réactif)', 'Baseline 2\n(Bayésien)', 'Baseline 3\n(Hybride)', 'Orchestrateur\n(L0-L3)']
    couts = [np.mean(resultats["L1"]["cout"]), np.mean(resultats["L2"]["cout"]), np.mean(resultats["HYB"]["cout"]), np.mean(resultats["ORC"]["cout"])]
    barres = plt.bar(architectures, couts, color=[couleurs['L1'], couleurs['L2'], couleurs['HYB'], couleurs['ORC']])
    plt.title("Figure 2 : Coût Computationnel Moyen", fontsize=14, fontweight='bold')
    plt.ylabel("Coût d'inférence")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    for barre in barres:
        yval = barre.get_height()
        plt.text(barre.get_x() + barre.get_width()/2, yval + 0.2, round(yval, 2), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.show()
