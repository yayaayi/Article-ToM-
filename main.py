import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# REPRODUCTIBILITÉ STRICTE (Fixation de la graine)
# ==========================================
np.random.seed(42)

# ==========================================
# HYPERPARAMÈTRES DE LA SIMULATION
# ==========================================
NOMBRE_RUNS = 30         # Méthode de Monte-Carlo pour l'analyse statistique
NOMBRE_EPISODES = 5000   # Nombre d'interactions par run
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
# BOUCLE PRINCIPALE DE SIMULATION (Monte-Carlo)
# ==========================================

if __name__ == "__main__":
    
    # Matrices pour stocker l'utilité cumulative à chaque épisode (pour la Figure 1)
    hist_utilite_L1 = np.zeros((NOMBRE_RUNS, NOMBRE_EPISODES))
    hist_utilite_L2 = np.zeros((NOMBRE_RUNS, NOMBRE_EPISODES))
    hist_utilite_HYB = np.zeros((NOMBRE_RUNS, NOMBRE_EPISODES))
    hist_utilite_ORC = np.zeros((NOMBRE_RUNS, NOMBRE_EPISODES))

    # Dictionnaires pour stocker les totaux par run (pour les statistiques du Tableau 3)
    stats_finales = {
        "L1": {"utilite": [], "cout_moyen": [], "faux_positifs": [], "l2_appels": [], "l3_vetos": []},
        "L2": {"utilite": [], "cout_moyen": [], "faux_positifs": [], "l2_appels": [], "l3_vetos": []},
        "HYB": {"utilite": [], "cout_moyen": [], "faux_positifs": [], "l2_appels": [], "l3_vetos": []},
        "ORC": {"utilite": [], "cout_moyen": [], "faux_positifs": [], "l2_appels": [], "l3_vetos": []}
    }

    modeles = ArchitecturesToM()
    print(f"Lancement de {NOMBRE_RUNS} simulations indépendantes de {NOMBRE_EPISODES} épisodes...")

    for run in range(NOMBRE_RUNS):
        env = SignalingGame()
        
        # Variables cumulatives pour le run en cours
        u_cumul_L1, u_cumul_L2, u_cumul_HYB, u_cumul_ORC = 0, 0, 0, 0
        couts_run = {"L1": 0, "L2": 0, "HYB": 0, "ORC": 0}
        fp_run = {"L1": 0, "L2": 0, "HYB": 0, "ORC": 0}
        l2_run = {"L1": 0, "L2": 0, "HYB": 0, "ORC": 0}
        veto_run = {"L1": 0, "L2": 0, "HYB": 0, "ORC": 0}

        for ep in range(NOMBRE_EPISODES):
            est_fraudeur, signal_ambigu = env.generer_interaction()

            # --- L1 ---
            act, cout, l2_trig, l3_v = modeles.baseline_1_reactive(signal_ambigu)
            u, fp = calculer_utilite(act, est_fraudeur)
            u_cumul_L1 += (u - cout)
            hist_utilite_L1[run, ep] = u_cumul_L1
            couts_run["L1"] += cout
            if fp: fp_run["L1"] += 1

            # --- L2 ---
            act, cout, l2_trig, l3_v = modeles.baseline_2_bayesienne(est_fraudeur)
            u, fp = calculer_utilite(act, est_fraudeur)
            u_cumul_L2 += (u - cout)
            hist_utilite_L2[run, ep] = u_cumul_L2
            couts_run["L2"] += cout
            if fp: fp_run["L2"] += 1
            if l2_trig: l2_run["L2"] += 1

            # --- HYBRIDE ---
            act, cout, l2_trig, l3_v = modeles.baseline_3_hybride(signal_ambigu, est_fraudeur)
            u, fp = calculer_utilite(act, est_fraudeur)
            u_cumul_HYB += (u - cout)
            hist_utilite_HYB[run, ep] = u_cumul_HYB
            couts_run["HYB"] += cout
            if fp: fp_run["HYB"] += 1
            if l2_trig: l2_run["HYB"] += 1

            # --- ORCHESTRATEUR ---
            act, cout, l2_trig, l3_v = modeles.orchestrateur_L0_L3(signal_ambigu, est_fraudeur)
            u, fp = calculer_utilite(act, est_fraudeur)
            u_cumul_ORC += (u - cout)
            hist_utilite_ORC[run, ep] = u_cumul_ORC
            couts_run["ORC"] += cout
            if fp: fp_run["ORC"] += 1
            if l2_trig: l2_run["ORC"] += 1
            if l3_v: veto_run["ORC"] += 1

        # Enregistrement des statistiques de fin de run
        for cle in ["L1", "L2", "HYB", "ORC"]:
            stats_finales[cle]["utilite"].append(eval(f"u_cumul_{cle}"))
            stats_finales[cle]["cout_moyen"].append(couts_run[cle] / NOMBRE_EPISODES)
            stats_finales[cle]["faux_positifs"].append((fp_run[cle] / NOMBRE_EPISODES) * 100)
            stats_finales[cle]["l2_appels"].append((l2_run[cle] / NOMBRE_EPISODES) * 100)
            if l2_run[cle] > 0:
                stats_finales[cle]["l3_vetos"].append((veto_run[cle] / l2_run[cle]) * 100)
            else:
                stats_finales[cle]["l3_vetos"].append(0)

    # ==========================================
    # AFFICHAGE DES RÉSULTATS DANS LA CONSOLE
    # ==========================================
    print("\n" + "="*110)
    print(f"{'ARCHITECTURE':<22} | {'UTILITÉ (± σ)':<18} | {'COÛT (± σ)':<16} | {'FAUX POS.':<10} | {'L2 INVOQUÉ':<12} | {'VETO L3 (sur L2)'}")
    print("="*110)

    for cle, nom in [("L1", "Baseline 1 (L1)"), ("L2", "Baseline 2 (L2)"), ("HYB", "Baseline 3 (Hybride)"), ("ORC", "Orchestrateur L0-L3")]:
        u_moy = np.mean(stats_finales[cle]["utilite"])
        u_std = np.std(stats_finales[cle]["utilite"])
        c_moy = np.mean(stats_finales[cle]["cout_moyen"])
        c_std = np.std(stats_finales[cle]["cout_moyen"])
        tx_fp = np.mean(stats_finales[cle]["faux_positifs"])
        tx_l2 = np.mean(stats_finales[cle]["l2_appels"])
        tx_veto = np.mean(stats_finales[cle]["l3_vetos"])
        
        veto_str = f"{tx_veto:>4.1f} %" if cle == "ORC" or cle == "L2" or cle == "HYB" else "N/A"
        if cle == "L1": tx_veto = 0 # Nettoyage affichage

        print(f"{nom:<22} | {u_moy:>6.0f} (± {u_std:>4.0f})     | {c_moy:>4.2f} (± {c_std:>4.2f})    | {tx_fp:>5.1f} %    | {tx_l2:>5.1f} %      | {veto_str}")
    print("="*110)

    # ==========================================
    # TEST STATISTIQUE DE WELCH
    # ==========================================
    t_stat, p_value = stats.ttest_ind(stats_finales["ORC"]["utilite"], stats_finales["L2"]["utilite"], equal_var=False)
    print(f"\n--- VALIDATION STATISTIQUE ---")
    print(f"Test t de Welch (Utilité ORC vs L2) : p-value = {p_value}")
    if p_value < 0.001:
        print("La différence d'utilité est HAUTEMENT SIGNIFICATIVE (***).")

    # ==========================================
    # GÉNÉRATION DES GRAPHIQUES
    # ==========================================
    print("\nGénération des figures en cours...")
    couleurs = {'L1': '#d62728', 'L2': '#2ca02c', 'HYB': '#ff7f0e', 'ORC': '#1f77b4'}

    # Calcul des courbes moyennes pour la Figure 1
    courbe_L1 = np.mean(hist_utilite_L1, axis=0)
    courbe_L2 = np.mean(hist_utilite_L2, axis=0)
    courbe_HYB = np.mean(hist_utilite_HYB, axis=0)
    courbe_ORC = np.mean(hist_utilite_ORC, axis=0)

    # --- FIGURE 1 : Utilité Cumulative ---
    plt.figure(figsize=(10, 6))
    plt.plot(courbe_L1, label="Baseline 1 (L1)", color=couleurs['L1'], linewidth=2)
    plt.plot(courbe_L2, label="Baseline 2 (L2)", color=couleurs['L2'], linewidth=2)
    plt.plot(courbe_HYB, label="Baseline 3 (Hybride)", color=couleurs['HYB'], linewidth=2, linestyle='-.')
    plt.plot(courbe_ORC, label="Orchestrateur (L0-L3)", color=couleurs['ORC'], linewidth=2, linestyle='--')
    
    plt.title(f"Figure 1 : Évolution de l'Utilité Cumulative (Moyenne sur {NOMBRE_RUNS} runs)", fontsize=14, fontweight='bold')
    plt.xlabel("Épisodes d'interaction")
    plt.ylabel("Utilité institutionnelle cumulative")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show() 

    # --- FIGURE 2 : Coût Moyen ---
    plt.figure(figsize=(8, 6))
    architectures = ['Baseline 1\n(Réactif)', 'Baseline 2\n(Bayésien)', 'Baseline 3\n(Hybride)', 'Orchestrateur\n(L0-L3)']
    couts_finaux = [np.mean(stats_finales["L1"]["cout_moyen"]), 
                    np.mean(stats_finales["L2"]["cout_moyen"]), 
                    np.mean(stats_finales["HYB"]["cout_moyen"]), 
                    np.mean(stats_finales["ORC"]["cout_moyen"])]
    
    barres = plt.bar(architectures, couts_finaux, color=[couleurs['L1'], couleurs['L2'], couleurs['HYB'], couleurs['ORC']])
    plt.title(f"Figure 2 : Coût Computationnel Moyen par Épisode", fontsize=14, fontweight='bold')
    plt.ylabel("Coût d'inférence")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    for barre in barres:
        yval = barre.get_height()
        plt.text(barre.get_x() + barre.get_width()/2, yval + 0.2, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.show()
