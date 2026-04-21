# Simulation : Orchestration Dynamique de la Théorie de l'Esprit (L0-L3)

Ce dépôt contient le code source de la simulation associée à l'article portant sur l'orchestration dynamique de la Théorie de l'Esprit (ToM) et la gouvernance algorithmique dans les Systèmes Multi-Agents (SMA).

L'objectif de ce code est de fournir une preuve empirique (reproductibilité totale) de l'avantage asymptotique de l'architecture d'orchestration proposée, en générant les résultats comparatifs détaillés dans l'article (Tableau des performances et taux de faux positifs).

## 1. Description de l'Environnement

La simulation implémente un jeu de signalement stratégique (*Signaling Game*) asymétrique utilisant le framework **PettingZoo**. 
L'environnement confronte un agent "Orchestrateur" à une population mixte :
* 80 % d'agents honnêtes (soumis à un bruit stochastique $P = 0.10$).
* 20 % d'agents fraudeurs (dissimulation stratégique $P = 0.20$).

L'Orchestrateur doit minimiser le coût computationnel tout en respectant une norme institutionnelle (présomption d'innocence) qui interdit les sanctions basées sur des preuves insuffisantes.

## 2. Structure du Dépôt

* `main.py` : Le script d'exécution principal qui lance les 5 000 épisodes de simulation, calcule les moyennes et génère les métriques.
* `environnement.py` : La définition de l'environnement de simulation et des règles de transition.
* `agents.py` : L'implémentation des différentes architectures testées :
  * **Baseline 1** : Agent réactif pur (L1).
  * **Baseline 2** : Agent bayésien pur (L2).
  * **Orchestrateur (L0-L3)** : Notre modèle avec déclenchement métacognitif (surprise) et gouvernance (confiance épistémique).

## 3. Prérequis et Installation

Pour exécuter cette simulation, vous devez disposer de Python 3.8+ et installer les bibliothèques suivantes :

```bash
pip install numpy pettingzoo matplotlib
