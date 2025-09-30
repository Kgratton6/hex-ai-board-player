Pour exécuter le code:

Premièrement il est nécessaire d'utiliser Python3.11
Une fois la bonne version de python installée, créer un environnement virtuel:
```bash
python -m venv venv
```
Et l'activer.

Ensuite il faut installer les requirements:
```bash
pip install -r requirements.txt
```

Pour lancer une partie, on peut utiliser les commandes présentées dans l'énoncé:

Partie humain contre humain:
```bash
python main_hex.py -t human_vs_human
```

Pour faire affronter 2 agents:
```bash
python main_hex.py -t local agent1.py agent2.py
```

Un agent aléatoire et un greedy sont fournis:
```bash
python main_hex.py -t local .\random_player_hex.py .\greedy_player_hex.py
```

Pour affronter humain contre agent:
```bash
python main_hex.py -t human_vs_computer agent.py
```

Pour affronter un agent d'un autre groupe:
Sur le PC hébergant le match:
```bash
python main_hex.py -t host_game -a <ip_adress> agent.py
```
Et l'équipe qu'on affronte devra lancer, avec l'IP de l'équipe qui héberge:
```bash
python main_hex.py -t connect -a <ip_adress> agent.py
```

## Règles
Le premier joueur joue les pions rouges, le second les bleus. Le joueur rouge doit relier par un chemin continu le haut et le bas du plateau, tandis que le bleu doit relier la gauche et la droite. 

A chaque tour, un joueur va poser une pièce de sa couleur sur une case vide du plateau.