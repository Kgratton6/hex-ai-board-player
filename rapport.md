# Rapport technique exhaustif — Agent Hex « MyPlayer »

Date: 2025-11-08  
Version: 1.1 (rapport unifié, sans préface séparée)  
Projet: INF8175 — Automne 2025

Résumé exécutif
- Ce document détaille intégralement la conception, la mise en œuvre, le style de jeu, le déroulement typique en partie, les choix d’architecture et les raisons associées de notre agent Hex « MyPlayer ».
- Nous conservons le vocabulaire technique (heuristique, ordonnancement, alpha–bêta, Zobrist, bridges, ziggurat, etc.) et l’expliquons clairement au fil du texte, avec renvoi direct vers le code source.
- L’agent équilibre trois axes: défendre (bloquer quand c’est urgent), construire (progresser vers sa connexion), et structurer (exploiter des motifs solides: bridges, ziggurat, rangs de bord). Le blocage n’est pas « systématisé »; c’est une composante parmi d’autres, arbitrée par l’évaluation et l’ordonnancement.

Références code principales (cliquables)
- Point d’entrée décision: [MyPlayer.compute_action()](my_player.py:79)
- Recherche (IDS + alpha–bêta): [MyPlayer._iterative_deepening_search()](my_player.py:290), [MyPlayer._alphabeta()](my_player.py:343)
- Table de transposition / Zobrist: [MyPlayer._init_zobrist()](my_player.py:53), [MyPlayer._zobrist_hash()](my_player.py:63)
- Évaluations: [MyPlayer._fast_static_eval()](my_player.py:1914), [MyPlayer._evaluate()](my_player.py:1891), [MyPlayer._root_order_score()](my_player.py:1976)
- Distance de connexion: [MyPlayer._connection_distance()](my_player.py:2252)
- Ordonnancement des coups: [MyPlayer._order_actions_with_spine()](my_player.py:488), [MyPlayer._order_actions_simple()](my_player.py:618)
- Filtrage des coups morts (P6): [MyPlayer._filter_dead_cells()](my_player.py:411)
- Must-block (gagne en 1): [MyPlayer._must_block_cells()](my_player.py:926)
- Bridges: [MyPlayer._bridge_creation_info()](my_player.py:1753), [MyPlayer._bridge_intrusion_replies()](my_player.py:1568), [MyPlayer._handle_bridge_replies()](my_player.py:1621)
- Ziggurat: [MyPlayer._ziggurat_specs_raw()](my_player.py:1144), [MyPlayer._ziggurat_intrusion_response()](my_player.py:1382)
- Réduction de bord (2→1, 3→2): [MyPlayer._edge_reduce_rank2_to_rank1_for_player()](my_player.py:1025), [MyPlayer._edge_reduce_for_player()](my_player.py:938)
- Émulation adverse / « front-cap »: [MyPlayer._best_opponent_moves()](my_player.py:701), [MyPlayer._adversary_emulation_response()](my_player.py:763), [MyPlayer._advance_two_forward()](my_player.py:723), [MyPlayer._forward_candidates()](my_player.py:744)
- Avancée forcée (après série de near-blocks): [MyPlayer._forced_advance2_if_needed()](my_player.py:640)

1. Contexte et objectifs
- Hex 14×14: jeu de connexion à deux joueurs. Rouge relie le haut au bas; Bleu relie la gauche à la droite. Aucune égalité (théorème de Nash).
- Objectif de l’agent: maximiser ses chances de victoire sous budget temps limité (≈ secondes par coup) et mémoire raisonnable, en privilégiant une stratégie intelligible, robuste et explicable.
- Principes directeurs:
  1) Priorité au « must-block » (si l’adversaire gagne au prochain coup).
  2) Construction proactive de notre chemin via motifs connus.
  3) Évaluation pilotée par la distance de connexion (pour nous et pour l’adversaire), avec pondération asymétrique et facteur d’urgence.
  4) Ordonnancement « intelligent » des coups pour réduire le coût de la recherche.
  5) Journalisation riche pour analyser les décisions.

2. Règles Hex en bref et implications techniques
- Placement alterné de pierres sur cases vides. Pierre posée = définitive.
- Victoire: relier ses deux bords. Géométrie hexagonale: 6 voisins par case (sauf bords).
- Implications:
  - Pas de captures, pas de « mobilité »: les tactiques reposent sur la forme (structures robustes) et la topologie du graphe.
  - Métrique-clé: « distance de connexion » du joueur i (combien de cases vides minimales à combler pour relier ses bords).

3. Architecture générale de l’agent
3.1 Classe et état interne
- L’agent est une classe: [MyPlayer](my_player.py:14).
- Paramètres clés: profondeur max (par défaut 4), flags de debug, logs « top-K », forçage d’avancée éventuelle.
- États internes (préfixés « _ » pour être exclus des journaux JSON Seahorse): compteur de nœuds, profondeur atteinte, deadline par coup, snapshot des positions, compteur near-blocks, table de transposition.

3.2 Pipeline décisionnel (ordre strict)
Déroulé dans [MyPlayer.compute_action()](my_player.py:79):
1) Must-block (gagne en 1): [MyPlayer._must_block_cells()](my_player.py:926). S’il existe, choisir le « meilleur » blocage via évaluation rapide.
2) Avancée forcée (si 3 near-blocks de suite): [MyPlayer._forced_advance2_if_needed()](my_player.py:640).
3) Réaction à intrusion de bridge: [MyPlayer._bridge_intrusion_replies()](my_player.py:1568) → [MyPlayer._handle_bridge_replies()](my_player.py:1621).
4) Émulation adverse (pivotal mais pas exclusive): [MyPlayer._adversary_emulation_response()](my_player.py:763).
5) Si plateau plus dense: Ziggurat (intrusions), Edge reduce 2→1, puis 3→2.
6) Sinon / en dernier recours: IDS (recherche) avec ordonnancement agressif et tie-break « bridge ».

Remarque d’équilibre
- Le bloc « émulation adverse » n’écrase pas les autres: c’est une étape parmi d’autres. La progression par motifs (bridges, ziggurat) et par bords (2→1, 3→2) pèse également dans la décision selon le contexte évalué.

4. Mesure centrale: distance de connexion (d_opp, d_my)
4.1 Définition
- Pour un joueur, c’est le nombre minimal de cases vides nécessaires pour relier ses deux bords (réseau de voies accessibles).
- Calculée par un Dijkstra 0–1: se déplacer vers une case à soi coûte 0 (déjà posée), vers une case vide coûte 1, vers une case adverse est interdit.
- Implémentation: [MyPlayer._connection_distance()](my_player.py:2252).

4.2 Pourquoi c’est utile
- d_opp (adversaire): plus il est grand, plus sa victoire est lointaine.
- d_my (moi): plus il est petit, plus je suis proche de gagner.
- Ces quantités se comparent coup par coup (avant/après) pour estimer l’impact des coups candidats.

5. Heuristique d’évaluation (H1) et évaluation rapide
5.1 Évaluation rapide (_fast_static_eval)
- [MyPlayer._fast_static_eval()](my_player.py:1914) combine des termes structurels:
  - Différence de pions (normalisée).
  - Centre (renforcé en ouverture): occuper la zone centrale donne souvent plus d’options.
  - Ancrage (proximité de mes bords cibles): [MyPlayer._edge_anchor_score()](my_player.py:2006).
  - Cluster (voisinage allié moyen): cohésion locale.
  - Présence sur les bords (0, 0.5, 1).
  - Bonus « bridge doux » si motifs de création/complétion existent (quand pas en must-block).
- Sortie bornée [-0.9, 0.9] pour stabilité.

5.2 Évaluation principale (H1)
- [MyPlayer._evaluate()](my_player.py:1891): 0.30·(éval rapide) + 0.70·(dist_term·urgence),
  - dist_term = (1.0·d_opp − 0.3·d_my)/n (pondération asymétrique).
  - urgence = 1 + 0.2·max(0, 10 − d_opp): plus l’adversaire est proche, plus on « pousse » d_opp dans l’évaluation.
- Lecture: on valorise surtout les coups qui « augmentent d_opp » (le font reculer), tout en gardant une part pour nos propres progrès (réduire d_my), et des termes structurels pour éviter des coups purement défensifs sans plan.

6. Ordonnancement des coups (ordering)
6.1 Racine (riche, orienté diagnostic)
- [MyPlayer._order_actions_with_spine()](my_player.py:488) classe les actions avant recherche:
  - Score de base (_root_order_score): [MyPlayer._root_order_score()](my_player.py:1976).
  - Δd_opp fort (avec urgence) + Δd_my léger.
  - Bonus « épine dorsale » (spine): chemin guidé bord→bord pour orienter les choix.
  - Bridges et gradient de distance: [MyPlayer._bridge_guided_bonus()](my_player.py:1833).
  - Bonus « front-cap » (coiffer le dernier coup adverse).
  - Pénalité si, en situation critique (d_opp faible), le coup n’augmente pas d_opp.
  - Logs top-K: score, d_opp1, deltas, front, distance à l’épine, etc.
- Idée: explorer d’abord les coups les plus prometteurs pour économiser du temps et activer mieux l’alpha–bêta.

6.2 Profondeur (léger)
- [MyPlayer._order_actions_simple()](my_player.py:618): score = _fast_static_eval + (terme distance simple × urgence).
- But: prioriser, sans coût élevé, en profondeur.

6.3 Pourquoi l’ordonnancement
- L’alpha–bêta coupe mieux si le bon coup est testé tôt. L’ordonnancement est donc une « heuristique d’exploration » qui rend la recherche plus efficace.

7. Filtrage des coups morts (P6)
- [MyPlayer._filter_dead_cells()](my_player.py:411) élimine des coups sans valeur stratégique immédiate, mais garde:
  - Must-block, trous de bridge (allié/adverse), centre raisonnable, front-cap, coups qui augmentent d_opp (≥+1) ou réduisent d_my.
- Quand tout est filtré: on revient à la liste originale (sécurité).

8. Motifs de construction et de robustesse
8.1 Bridges (arches)
- Intuition: deux pierres alliées non adjacentes forment un « losange » avec deux trous potentiels. Remplir l’un, c’est souvent garder l’autre libre: structure robuste.
- Détection & réaction:
  - Détection d’intrusion: [MyPlayer._bridge_intrusion_replies()](my_player.py:1568) repère si l’adversaire s’insère entre deux de nos pierres en créant une « faiblesse ».
  - Réponse: [MyPlayer._handle_bridge_replies()](my_player.py:1621) choisit le coup qui répare/sécurise sans dégrader notre distance.
  - Information « crée/complète »: [MyPlayer._bridge_creation_info()](my_player.py:1753).
- Intérêt: consolider nos couloirs tout en gardant des options.

8.2 Ziggurat (pyramide de bord)
- Petit gabarit 9-cases (4–3–2) près d’un bord, selon la couleur (vertical pour Rouge, horizontal pour Bleu).
- Spécification et validation: [MyPlayer._ziggurat_specs_raw()](my_player.py:1144) / [MyPlayer._ziggurat_specs()](my_player.py:1214).
- Réponse à intrusion (exactement une pierre adverse dans le « carrier »): [MyPlayer._ziggurat_intrusion_response()](my_player.py:1382), qui choisit une « sortie » A ou B selon le côté attaqué.
- Intérêt: fournir un tremplin stable pour achever la connexion par le bord.

8.3 Progression aux bords (réductions 3→2 et 2→1)
- 3→2: [MyPlayer._edge_reduce_for_player()](my_player.py:938).
- 2→1: [MyPlayer._edge_reduce_rank2_to_rank1_for_player()](my_player.py:1025).
- Règles (en bref):
  - Ne pas « pousser » si la composante touche déjà le bord cible.
  - Respecter l’adjacence à la composante « rang 2 ».
  - Vérifier que la distance d_my ne se dégrade pas.
- Activation en milieu/fin de partie (plateau plus dense), pour ne pas perturber l’ouverture.

9. Émulation de l’adversaire et « front-cap »
9.1 Idée
- Simuler les meilleurs coups probables de l’adversaire (ceux qui réduisent le plus sa distance d_opp): [MyPlayer._best_opponent_moves()](my_player.py:701).
- Comparer nos réponses candidates:
  - Blocage direct p_best (la case jugée « meilleure » pour lui).
  - « Front-cap » (coiffer son dernier coup dans sa direction naturelle).
  - Avancer nous-mêmes (advance2) si opportun.
- Choisir ce qui maximise d_opp après notre coup (tie → front-cap).

9.2 Détails
- Fonction pivot: [MyPlayer._adversary_emulation_response()](my_player.py:763).
- Seuil « menace forte »: si Δ≥2 (son meilleur coup réduit d_opp d’au moins 2) ou si d_opp≤8 (il est proche de connecter), on compare finement front-cap vs blocage p_best et prend l’argmax d_opp (égalité → front-cap).
- Sinon, tenter l’advance2 (avancer de 1/2 en « façade » pour construire un mur/progression), mais seulement si après notre coup d_opp augmente au moins autant que bloquer p_best — sinon fallback blocage.

9.3 Avancée forcée (après série de near-blocks)
- [MyPlayer._forced_advance2_if_needed()](my_player.py:640) déclenche si on a trop « subi » sans progresser (3 blocages consécutifs).
- Corridor fallback: s’il n’y a pas de +2 direct, tenter un +1 dans un corridor raisonnable.
- Sécurité: si, après notre avance, l’adversaire a un must-block gagnant, on annule (sauf si le flag autorise la prise de risque).

Remarque d’équilibre (bis)
- Le blocage n’est ni unique ni systématique. L’agent l’emploie:
  - En priorité absolue quand c’est vital (must-block).
  - En « menace forte » si le rapport coût/bénéfice est clair.
  - Sinon, il préfère avancer ou structurer (bridges, ziggurat, bords) si ces actions offrent un meilleur futur (d_opp plus élevé, d_my plus faible, structure plus solide).

10. Moteur de recherche (IDS + alpha–bêta + TT)
10.1 IDS (Iterative Deepening)
- [MyPlayer._iterative_deepening_search()](my_player.py:290).
- Recherche par paliers 1..Dmax; interrompt proprement si la deadline approche; réutilise le meilleur du palier précédent comme premier à tester (aspiration).

10.2 Alpha–Bêta
- [MyPlayer._alphabeta()](my_player.py:343).
- Cadre minimax avec coupe de branches quand on sait qu’un coup ne peut pas améliorer le résultat.
- Plus l’ordonnancement est pertinent, plus les coupes sont efficaces.

10.3 Table de transposition (mémo d’états)
- Éviter de réévaluer le même plateau: [MyPlayer._zobrist_hash()](my_player.py:63) donne une empreinte (Zobrist).
- La TT stocke valeur, profondeur, borne (EXACT/LOWER/UPPER), meilleur coup; on réutilise si profondeur suffisante.

10.4 Pourquoi ce trio?
- L’IDS garantit une « meilleure solution connue » même en cas de coupure de temps.
- L’alpha–bêta maintient le coût sous contrôle.
- La TT exploite les répétitions de sous-arbres.

11. Gestion du temps, logs et robustesse
11.1 Temps
- Budget par coup: 2% du temps restant, borné à [0.3s, 1.5s]: [MyPlayer.compute_action()](my_player.py:86).
- Chaque boucle lourde vérifie la deadline; en dernier ressort, on renvoie l’évaluation rapide au lieu d’aller plus loin.

11.2 Logs
- Indicateurs lisibles:
  - [must] menaces gagnantes en 1.
  - [forced] déclenchement/annulation d’advance2 forcé.
  - [opp] meilleurs coups adverses simulés (d1, Δ).
  - [order] top-K ordonnancé avec scores et détails.
  - [choice] deltas de distances après le coup choisi.
  - Bridge posé / reply.
- Utiles pour comprendre « pourquoi » un coup a été choisi et pour affiner les heuristiques.

11.3 Robustesse
- Vérification de légalité des actions renvoyées (conversion via LightAction).
- Fallbacks: si une ligne échoue (ex.: pas d’advance possible), on retombe sur blocage p_best ou recherche IDS.
- Exceptions capturées ponctuellement pour ne pas interrompre le cycle.

12. Déroulement typique d’un tour (pas-à-pas commenté)
Supposons: Bleu joue, Rouge vient de jouer « H8 ».
1) Déduction du dernier coup adverse: [MyPlayer._deduce_last_opponent_move()](my_player.py:889). Log: « last_opp=H8 ».
2) Must-block? [MyPlayer._must_block_cells()](my_player.py:926). Si oui, liste de cases à bloquer + choix du meilleur via [MyPlayer._choose_best_from_candidates()](my_player.py:1647). Sinon continuer.
3) Série de near-blocks? Si oui, tenter [MyPlayer._forced_advance2_if_needed()](my_player.py:640): logs [forced].
4) Intrusion au bridge? [MyPlayer._bridge_intrusion_replies()](my_player.py:1568) → [MyPlayer._handle_bridge_replies()](my_player.py:1621). Si choisi: log « bridge_intrusion reply=... ».
5) Émulation adverse: [MyPlayer._best_opponent_moves()](my_player.py:701) donne une liste triée, log [opp] avec d0, d1, Δ.
   - Si menace forte, comparer front-cap vs blocage p_best (simuler d_opp après nos deux options). Égalité → front-cap.
   - Sinon, tenter advance2 (simuler d_opp après) ; si pas concluant, fallback blocage p_best.
   - Log [opp->play] avec « reason » (front_cap / block_best / advance2 / fallback_block).
6) Si plateau densifié: ziggurat → edge 2→1 → edge 3→2. Logs « ziggurat_response=... », « edge_reduce_... ».
7) Sinon: IDS. Log [order] top-K, puis [choice] avec Δ distances du coup final.

Lecture des logs clés:
- [order] liste les K meilleurs coups après ordonnancement racine, avec:
  - sc=score total, d_opp1=distance adverse après le coup, Δopp=(d_opp1−d_opp0)/n, Δmy, front (1 si front-cap), spine=distance à l’épine, bbonus=bonus bridge, penNoRaise (pénalité si d_opp n’augmente pas en situation critique).
- [choice] synthétise l’effet: d_opp:7→8 (Δ+1) signifie « l’adversaire a 1 pas de plus à fournir ».

13. Choix d’implémentation et raisons
- Distance de connexion comme boussole: mesure simple, stable, indépendante des motifs, adaptée au graphe hexagonal.
- Pondération asymétrique (w_opp > w_my) et urgence: encourage à « tenir la digue » quand l’adversaire approche, sans oublier de construire.
- Bridges et ziggurat: intégrer des motifs classiques connus pour éviter de tout attendre de la recherche brute.
- Réductions de bord: aide à « concrétiser » les connexions en fin de partie.
- Émulation adverse: couvrir un cas visuel fréquent (« couloir droit ») tout en respectant l’arbitrage global; ce n’est pas une règle brute, mais une comparaison simulée (d_opp après coup).
- Avancée forcée: éviter l’oscillation « bloquer sans fin »; injecter des percées proactives mais sécurisées.

14. Expérimentation et résultats (mode d’emploi)
- Lancer des matchs:
  - Humain vs IA: `python main_hex.py -t human_vs_computer my_player.py`
  - IA vs greedy: `python main_hex.py -t local my_player.py greedy_player_hex.py`
  - Sans GUI (logs JSON): `python main_hex.py -t local my_player.py random_player_hex.py -r -g`
- Indicateurs:
  - Taux de victoire vs agents de base (random/greedy).
  - Δd_opp moyen par coup choisi (plus c’est positif en défense, mieux on freine).
  - Proportion des raisons [opp->play] (front_cap/block_best/advance2): équilibre attaque/défense.
  - Fréquence des bridges posés/complétés.

Observations attendues
- Contre random/greedy: l’IA bloque les « couloirs faciles » et construit des arches/bords en parallèle.
- En milieu de partie: usage croissant de ziggurat/edge-reduce quand des structures s’installent.
- Les tie-break « bridge » (à valeur quasi-égale) favorisent de « bons schémas » plutôt que des gains purement locaux.

15. Limites et pistes d’amélioration
- Profondeur modeste (4) sous contrainte de temps: certaines tactiques à horizon long échappent aux palier courts.
- Heuristique « corridor/avant » reste géométriquement approximative (hex vs grille régulière).
- Quiescence minimale: on pourrait prolonger sélectivement si un coup « agite » beaucoup la distance (grosse baisse de d_opp).
- Cache distances: mémo par (hash état, couleur) dans [MyPlayer._connection_distance()](my_player.py:2252) pour accélérer des scénarios denses.
- Apprentissage des pondérations (w_opp, w_my, seuils d_opp, Δ): grid search ou optimisation bayésienne via auto-jeu.

16. Guide d’exécution / paramétrage
- Installation
  - `python -m venv venv`
  - (Windows) `.\venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
- Lancement types
  - GUI local: `python main_hex.py -t local my_player.py greedy_player_hex.py`
  - Humain vs IA: `python main_hex.py -t human_vs_computer my_player.py`
- Paramètres utiles
  - Profondeur: ctor [MyPlayer.__init__()](my_player.py:29)
  - Logs: [_debug](my_player.py:41), [_root_log_top_k](my_player.py:42)
  - Forcing risqué advance2: [_force_advance2_even_if_threat](my_player.py:45)

17. Glossaire technique (conservant le jargon, expliqué)
- Heuristique: « règle de pouce » mathématisée pour évaluer/coter rapidement une position (moins cher qu’une recherche complète).
- Ordonnancement (ordering): classement des coups à tester en priorité, pour aider la recherche à couper des branches inutiles.
- Alpha–bêta: technique qui écarte des sous-arbres qui ne peuvent pas améliorer le résultat, accélérant le minimax.
- IDS (Iterative Deepening Search): recherche par paliers croissants, toujours interrompable avec la meilleure solution connue.
- Zobrist: hachage d’un plateau (grande empreinte binaire) en XORant des constantes aléatoires associées à (couleur, case), plus le trait.
- Table de transposition (TT): dictionnaire de plateaux déjà évalués (par leur Zobrist), pour mémo-iser les résultats.
- Bridge (arche): motif robuste à deux trous; en combler un laisse de la flexibilité.
- Ziggurat (pyramide de bord): gabarit 9-cases qui sécurise une pente vers le bord.
- Front-cap (coiffe): poser juste « devant » la dernière pierre adverse dans sa direction d’avancée, pour refermer un couloir.
- Edge reduce (réduction de bord): déplacer la « pression » d’un anneau 3→2 puis 2→1 pour finaliser une connexion.

18. Cartographie fine code → concepts
- Pipeline complet: [MyPlayer.compute_action()](my_player.py:79)
- Must-block: [MyPlayer._must_block_cells()](my_player.py:926) (utilisé en tête du pipeline)
- Avancée forcée: [MyPlayer._forced_advance2_if_needed()](my_player.py:640)
- Bridges:
  - Détection structure: [MyPlayer._bridge_creation_info()](my_player.py:1753)
  - Réaction intrusion: [MyPlayer._bridge_intrusion_replies()](my_player.py:1568), [MyPlayer._handle_bridge_replies()](my_player.py:1621)
- Ziggurat:
  - Specs/validation: [MyPlayer._ziggurat_specs_raw()](my_player.py:1144), [MyPlayer._ziggurat_specs()](my_player.py:1214)
  - Réponse intrusion: [MyPlayer._ziggurat_intrusion_response()](my_player.py:1382)
- Bords: [MyPlayer._edge_reduce_for_player()](my_player.py:938), [MyPlayer._edge_reduce_rank2_to_rank1_for_player()](my_player.py:1025)
- Émulation adverse:
  - Top coups adverses: [MyPlayer._best_opponent_moves()](my_player.py:701)
  - Réponse: [MyPlayer._adversary_emulation_response()](my_player.py:763)
  - Avance « en façade »: [MyPlayer._advance_two_forward()](my_player.py:723), [MyPlayer._forward_candidates()](my_player.py:744)
- Ordonnancement:
  - Racine riche: [MyPlayer._order_actions_with_spine()](my_player.py:488)
  - Profondeur légère: [MyPlayer._order_actions_simple()](my_player.py:618)
- Évaluation:
  - Rapide: [MyPlayer._fast_static_eval()](my_player.py:1914)
  - H1: [MyPlayer._evaluate()](my_player.py:1891)
  - Score racine: [MyPlayer._root_order_score()](my_player.py:1976)
- Distance: [MyPlayer._connection_distance()](my_player.py:2252)
- Recherche:
  - IDS: [MyPlayer._iterative_deepening_search()](my_player.py:290)
  - Alpha–bêta + TT: [MyPlayer._alphabeta()](my_player.py:343), [MyPlayer._zobrist_hash()](my_player.py:63)

Conclusion
Notre agent « MyPlayer » combine une évaluation « distance-driven » nuancée (pas uniquement défensive), un ordonnancement riche, des motifs de structure (bridges, ziggurat, bords) et une étape d’émulation de l’adversaire qui compare objectivement blocage et coiffure. Le pipeline est priorisé de manière sûre (must-block d’abord), tout en laissant de la place aux progrès structurés et à la recherche générale. Les journaux offrent une traçabilité fine des raisons de chaque coup. Des marges d’amélioration existent (quiescence ciblée, cache de distances, apprentissage des pondérations), mais l’architecture actuelle réalise déjà un bon équilibre entre défense, construction et efficacité de calcul.
