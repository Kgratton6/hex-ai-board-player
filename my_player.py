from typing import Optional, cast, Any
import random
import time
import heapq

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from game_state_hex import GameStateHex
from board_hex import BoardHex


class MyPlayer(PlayerHex):
    """
    Agent Hex avec:
    - P0-P4: Alpha-bêta, H1 (distance+centre), ordering racine, bridge réactif
    - P3.2: Épine dorsale (spine) + bonus bridges guidés + gradient
    - P5: Réduction de bord 3→2
    - P6: Filtrage dead cells
    - P7: Must-play de couloir
    - P9: IDS (Iterative Deepening Search)
    - P10: Table de transposition (Zobrist)
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer", max_depth: int = 4):
        super().__init__(piece_type, name)
        self.max_depth = max(1, int(max_depth))
        
        # Stats (préfixées _ pour ne pas apparaître dans JSON)
        self._nodes_visited = 0
        self._depth_reached = 0
        self._deadline: float = float("inf")
        self._prev_positions: set[tuple[int, int]] = set()
        
        # Table de transposition (Zobrist)
        self._tt: dict[int, tuple[float, int, str, Optional[tuple[int,int]]]] = {}
        self._zobrist_table: dict[tuple[str, int, int], int] = {}
        self._zobrist_side: dict[int, int] = {}
        self._init_zobrist()

    def _init_zobrist(self):
        """Initialise la table Zobrist pour le hashing d'états"""
        random.seed(42)  # Pour reproductibilité
        for piece_type in ["R", "B"]:
            for i in range(14):
                for j in range(14):
                    self._zobrist_table[(piece_type, i, j)] = random.getrandbits(64)
        self._zobrist_side[0] = random.getrandbits(64)
        self._zobrist_side[1] = random.getrandbits(64)

    def _zobrist_hash(self, state: GameStateHex) -> int:
        """Calcule le hash Zobrist de l'état"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        h = 0
        for (i, j), piece in env.items():
            try:
                pt = piece.get_type()
                h ^= self._zobrist_table[(pt, i, j)]
            except Exception:
                pass
        # XOR avec le joueur courant
        player_idx = 0 if state.next_player.get_id() == state.players[0].get_id() else 1
        h ^= self._zobrist_side[player_idx]
        return h

    def compute_action(self, current_state: GameState, remaining_time: int = 1_000_000_000, **kwargs) -> Action:  # type: ignore[override]
        """Sélectionne une action via IDS + alpha-bêta avec toutes les optimisations"""
        start = time.perf_counter()
        self._nodes_visited = 0
        self._depth_reached = 0
        self._tt.clear()  # Clear TT chaque coup (ou garder si mémoire suffisante)
        
        # Budget temps: 2% du temps restant, borné [0.3s, 1.5s]
        try:
            rt = float(remaining_time)
        except Exception:
            rt = 900.0
        per_move_budget = min(1.5, max(0.3, 0.02 * rt))
        self._deadline = start + per_move_budget
        
        state = current_state
        st_hex = cast(GameStateHex, state)
        rep_board = cast(BoardHex, st_hex.get_rep())
        env = rep_board.get_env()
        n = rep_board.get_dimensions()[0]
        
        # Déduire le dernier coup adverse
        curr_pos_set = set(env.keys())
        last_pos = self._deduce_last_opponent_move(st_hex, curr_pos_set)
        
        # Log du dernier coup
        if last_pos:
            print(f"[MyPlayer] last_opp={self._pos_to_an(last_pos, n)}")

        # PHASE 1: Must-block (gagne en 1 coup)
        threats = self._must_block_cells(st_hex)
        if threats:
            actions = list(state.get_possible_light_actions())
            candidates = [a for a in actions if a.data.get("position") in threats]
            if candidates:
                best_block = self._choose_best_from_candidates(st_hex, candidates)
                if best_block:
                    print(f"[MyPlayer] must_block size={len(candidates)}")
                    # Log si le coup posé forme un bridge (informationnelle)
                    pos = best_block.data.get("position")
                    if isinstance(pos, tuple):
                        self._log_bridge_decision(st_hex, pos, source="must_block")
                    self._update_prev_positions(curr_pos_set, best_block)
                    return best_block
        
        # PHASE 2: Bridge réactif
        replies = self._bridge_intrusion_replies(st_hex, self.piece_type, last_pos)
        if replies:
            actions = list(state.get_possible_light_actions())
            best_bridge = self._handle_bridge_replies(st_hex, replies, actions)
            if best_bridge:
                self._update_prev_positions(curr_pos_set, best_bridge)
                return best_bridge
        
        # PHASE 3: Ziggurat - réponse aux intrusions dans nos templates
        ziggurat_response = self._ziggurat_intrusion_response(st_hex, self.piece_type, last_pos)
        if ziggurat_response:
            actions = list(state.get_possible_light_actions())
            candidates = [a for a in actions if a.data.get("position") == ziggurat_response]
            if candidates:
                best_zig = candidates[0]
                print(f"[MyPlayer] ziggurat_response={self._pos_to_an(ziggurat_response, n)}")
                # Log si le coup posé forme un bridge (informationnelle)
                pos = best_zig.data.get("position")
                if isinstance(pos, tuple):
                    self._log_bridge_decision(st_hex, pos, source="ziggurat")
                self._update_prev_positions(curr_pos_set, best_zig)
                return best_zig

        # PHASE 4-5: Edge reduce (différés jusqu'à la fin d'ouverture)
        total_stones = len(env)
        if total_stones >= 10:
            # PHASE 4: Edge reduce (réduction 2→1 vers bord)
            edge21_moves = self._edge_reduce_rank2_to_rank1_for_player(st_hex, self.piece_type)
            if edge21_moves:
                actions = list(state.get_possible_light_actions())
                candidates = [a for a in actions if a.data.get("position") in edge21_moves]
                if candidates:
                    best_edge21 = self._choose_best_from_candidates(st_hex, candidates)
                    if best_edge21:
                        pos = best_edge21.data.get("position")
                        if isinstance(pos, tuple):
                            print(f"[MyPlayer] edge_reduce_2to1={self._pos_to_an(pos, n)}")
                            # Log si le coup posé forme un bridge (informationnelle)
                            if isinstance(pos, tuple):
                                self._log_bridge_decision(st_hex, pos, source="edge_2to1")
                            self._update_prev_positions(curr_pos_set, best_edge21)
                            return best_edge21

            # PHASE 5: Edge reduce (réduction 3→2)
            edge_moves = self._edge_reduce_for_player(st_hex, self.piece_type)
            if edge_moves:
                actions = list(state.get_possible_light_actions())
                candidates = [a for a in actions if a.data.get("position") in edge_moves]
                if candidates:
                    best_edge = self._choose_best_from_candidates(st_hex, candidates)
                    if best_edge:
                        pos = best_edge.data.get("position")
                        if isinstance(pos, tuple):
                            print(f"[MyPlayer] edge_reduce_3to2={self._pos_to_an(pos, n)}")
                            # Log si le coup posé forme un bridge (informationnelle)
                            if isinstance(pos, tuple):
                                self._log_bridge_decision(st_hex, pos, source="edge_3to2")
                            self._update_prev_positions(curr_pos_set, best_edge)
                            return best_edge

        # PHASE 6: (supprimé) ancien edge double-threat; remplacé prochainement par le module Blocking
        # PHASE 6: IDS + Alpha-bêta avec toutes les optimisations
        best_action = self._iterative_deepening_search(st_hex, last_pos)
        
        # Tie-break racine en faveur d'un bridge (epsilon 0.02), si pas de must-block
        try:
            epsilon = 0.02
            if not self._must_block_cells(st_hex):
                actions_all = list(state.get_possible_light_actions())
                child_best = cast(GameStateHex, st_hex.apply_action(best_action))
                best_val_est = self._evaluate(child_best)
                prefer = None
                prefer_val = float("-inf")
                prefer_kind: Optional[str] = None
                # Préférer d'abord 'create' puis la meilleure valeur
                def prio_tuple(kind: Optional[str], val: float) -> tuple[int, float]:
                    return (1 if kind == "create" else 0, val)
                for a in actions_all:
                    pos_a = a.data.get("position")
                    if not (isinstance(pos_a, tuple) and len(pos_a) == 2):
                        continue
                    child_a = cast(GameStateHex, st_hex.apply_action(a))
                    v = self._evaluate(child_a)
                    if v + epsilon < best_val_est:
                        continue
                    kind = self._classify_bridge_move(st_hex, pos_a)
                    if kind is None:
                        continue
                    if prefer is None or prio_tuple(kind, v) > prio_tuple(prefer_kind, prefer_val):
                        prefer = a
                        prefer_val = v
                        prefer_kind = kind
                if prefer is not None and prefer != best_action:
                    rep = cast(BoardHex, st_hex.get_rep())
                    n = rep.get_dimensions()[0]
                    ppos = prefer.data.get("position")
                    if isinstance(ppos, tuple):
                        print(f"[MyPlayer] bridge_tiebreak={self._pos_to_an(ppos, n)} kind={prefer_kind} eps={epsilon}")
                    best_action = prefer
        except Exception:
            pass
        
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"[MyPlayer] nodes={self._nodes_visited} time_ms={elapsed_ms:.1f} depth={self._depth_reached}")
        
        # Log si le coup final forme un bridge (informationnelle)
        pos = best_action.data.get("position")
        if isinstance(pos, tuple):
            self._log_bridge_decision(st_hex, pos, source="ids")
        self._update_prev_positions(curr_pos_set, best_action)
        return best_action

    def _iterative_deepening_search(self, state: GameStateHex, last_pos: Optional[tuple[int,int]]) -> LightAction:
        """IDS: recherche itérative en profondeur croissante"""
        actions = list(state.get_possible_light_actions())
        if not actions:
            raise RuntimeError("Aucune action légale disponible.")
        
        # Filtrage dead cells (P6)
        filtered_actions = self._filter_dead_cells(state, actions, last_pos)
        if not filtered_actions:
            filtered_actions = actions
        
        # Ordonnancement initial
        ordered_actions = self._order_actions_with_spine(state, filtered_actions, last_pos)
        
        best_action = ordered_actions[0]
        best_value = float("-inf")
        
        # IDS: profondeur croissante
        for depth in range(1, self.max_depth + 1):
            if time.perf_counter() >= self._deadline:
                break
            
            self._depth_reached = depth
            temp_best = None
            temp_value = float("-inf")
            alpha, beta = float("-inf"), float("inf")
            
            # Réordonner avec le meilleur coup du palier précédent en tête
            if best_action and best_action in ordered_actions:
                ordered_actions.remove(best_action)
                ordered_actions.insert(0, best_action)
            
            for action in ordered_actions:
                if time.perf_counter() >= self._deadline:
                    break
                
                child = cast(GameStateHex, state.apply_action(action))
                value = self._alphabeta(child, depth - 1, alpha, beta, False)
                
                if value > temp_value:
                    temp_value = value
                    temp_best = action
                
                alpha = max(alpha, temp_value)
                if alpha >= beta:
                    break
            
            if temp_best is not None:
                best_action = temp_best
                best_value = temp_value
        
        return best_action

    def _alphabeta(self, state: GameStateHex, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Alpha-bêta avec table de transposition et quiescence"""
        if time.perf_counter() > self._deadline:
            return self._fast_static_eval(state)
        
        self._nodes_visited += 1
        
        # Vérifier TT
        zh = self._zobrist_hash(state)
        if zh in self._tt:
            tt_value, tt_depth, tt_flag, _ = self._tt[zh]
            if tt_depth >= depth:
                if tt_flag == "EXACT":
                    return tt_value
                elif tt_flag == "LOWER" and tt_value >= beta:
                    return tt_value
                elif tt_flag == "UPPER" and tt_value <= alpha:
                    return tt_value
        
        # Terminal
        if state.is_done():
            return float(self._utility(state))
        
        # Coupe
        if depth <= 0:
            # Quiescence simple: si le dernier coup était une intrusion tactique, prolonger d'1
            return float(self._evaluate(state))
        
        my_turn = (state.next_player.get_id() == self.get_id())
        actions = list(state.get_possible_light_actions())
        
        # Ordonnancement en profondeur (simplifié)
        ordered_actions = self._order_actions_simple(state, actions)
        
        if maximizing or my_turn:
            value = float("-inf")
            best_move = None
            for action in ordered_actions:
                child = cast(GameStateHex, state.apply_action(action))
                child_value = self._alphabeta(child, depth - 1, alpha, beta, False)
                if child_value > value:
                    value = child_value
                    best_move = action.data.get("position")
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            
            # Stocker dans TT
            flag = "EXACT" if value > alpha and value < beta else ("LOWER" if value >= beta else "UPPER")
            self._tt[zh] = (value, depth, flag, best_move)
            return value
        else:
            value = float("inf")
            best_move = None
            for action in ordered_actions:
                child = cast(GameStateHex, state.apply_action(action))
                child_value = self._alphabeta(child, depth - 1, alpha, beta, True)
                if child_value < value:
                    value = child_value
                    best_move = action.data.get("position")
                beta = min(beta, value)
                if alpha >= beta:
                    break
            
            flag = "EXACT" if value > alpha and value < beta else ("UPPER" if value <= alpha else "LOWER")
            self._tt[zh] = (value, depth, flag, best_move)
            return value

    def _filter_dead_cells(self, state: GameStateHex, actions: list[LightAction], last_pos: Optional[tuple[int,int]]) -> list[LightAction]:
        """P6: Filtrage des coups morts (coins, zones scellées)"""
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
        
        # Conserver les coups tactiques
        threats = self._must_block_cells(state)
        opp_type = "B" if self.piece_type == "R" else "R"
        opp_bridge_holes = self._bridge_holes(state, opp_type)
        my_bridge_holes = self._bridge_holes(state, self.piece_type)
        
        curr_dist = self._connection_distance(state, self.piece_type)
        opp_dist = self._connection_distance(state, opp_type)
        
        filtered = []
        for a in actions:
            pos = a.data.get("position")
            if not (isinstance(pos, tuple) and len(pos) == 2):
                continue
            
            i, j = pos
            
            # Garder si tactique
            if pos in threats or pos in opp_bridge_holes or pos in my_bridge_holes:
                filtered.append(a)
                continue
            
            # Garder si zone centrale (pas dans les coins extrêmes)
            if 2 <= i < n-2 and 2 <= j < n-2:
                filtered.append(a)
                continue
            
            # Garder si améliore notre distance ou dégrade la leur
            child = cast(GameStateHex, state.apply_action(a))
            new_my_dist = self._connection_distance(child, self.piece_type)
            new_opp_dist = self._connection_distance(child, opp_type)
            
            if new_my_dist < curr_dist or new_opp_dist > opp_dist:
                filtered.append(a)
                continue
        
        return filtered if filtered else actions

    def _order_actions_with_spine(self, state: GameStateHex, actions: list[LightAction], last_pos: Optional[tuple[int,int]]) -> list[LightAction]:
        """P3.2: Ordonnancement avec épine dorsale, bridges, gradient"""
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
        
        # Calcul de l'épine dorsale
        spine = self._guiding_spine(state, self.piece_type)
        
        # Pré-calculs
        opp_type = "B" if self.piece_type == "R" else "R"
        opp_bridge_holes = self._bridge_holes(state, opp_type)
        curr_dist = self._connection_distance(state, self.piece_type)
        border_reduce = self._border_reduce_candidates_from_actions(state, self.piece_type, curr_dist, actions)
        
        # Score chaque action
        scored = []
        for a in actions:
            pos = a.data.get("position")
            if not (isinstance(pos, tuple) and len(pos) == 2):
                continue
            
            # Éval de base
            child = cast(GameStateHex, state.apply_action(a))
            score = self._root_order_score(child)
            
            # Bonus épine + bridges + gradient (P3.2)
            spine_bonus = self._bridge_guided_bonus(state, pos, spine)
            score += spine_bonus
            
            # Bonus adjacence
            adj = self._friendly_neighbors_count(state, pos, self.piece_type)
            score += 0.18 * (adj / 6.0)
            
            # Bonus réduction bord
            if pos in border_reduce:
                score += 0.12
            
            # Bonus si création/complétion d'un ziggurat
            zig_bonus = self._ziggurat_formation_bonus(state, pos)
            score += zig_bonus
            
            # Bonus biais de flanc
            if last_pos:
                if self.piece_type == "B":
                    if (last_pos[1] <= 1 and pos[1] == 1) or (last_pos[1] >= n-2 and pos[1] == n-2):
                        score += 0.06
                else:
                    if (last_pos[0] <= 1 and pos[0] == 1) or (last_pos[0] >= n-2 and pos[0] == n-2):
                        score += 0.06
            
            # Pénalité trous adverses (sauf si tactique)
            if pos in opp_bridge_holes:
                child_tmp = cast(GameStateHex, state.apply_action(a))
                if self._utility(child_tmp) != 1:
                    score -= 0.12
            
            scored.append((a, score))
        
        scored.sort(key=lambda t: t[1], reverse=True)
        return [a for a, _ in scored]

    def _order_actions_simple(self, state: GameStateHex, actions: list[LightAction]) -> list[LightAction]:
        """Ordonnancement simplifié pour les niveaux profonds"""
        scored = []
        for a in actions:
            child = cast(GameStateHex, state.apply_action(a))
            score = self._fast_static_eval(child)
            scored.append((a, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [a for a, _ in scored]

    # ========== HELPERS TACTIQUES ==========

    def _deduce_last_opponent_move(self, state: GameStateHex, curr_pos_set: set[tuple[int,int]]) -> Optional[tuple[int,int]]:
        """Déduit le dernier coup adverse par différence d'états"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        
        try:
            prev = getattr(self, "_prev_positions", set())
            added_list = list(curr_pos_set - prev)
            opp_type = "B" if self.piece_type == "R" else "R"
            
            opp_added = []
            for pos in added_list:
                p = env.get(pos)
                if p is not None and p.get_type() == opp_type:
                    opp_added.append(pos)
            
            if len(opp_added) == 1:
                return opp_added[0]
            elif len(opp_added) > 1:
                return opp_added[0]
        except Exception:
            pass
        return None

    def _update_prev_positions(self, curr_pos_set: set[tuple[int,int]], action: LightAction):
        """Met à jour le snapshot de positions après notre coup"""
        try:
            pos = action.data.get("position")
            if isinstance(pos, tuple) and len(pos) == 2:
                snap = set(curr_pos_set)
                snap.add(pos)
                self._prev_positions = snap
            else:
                self._prev_positions = curr_pos_set
        except Exception:
            self._prev_positions = curr_pos_set

    def _must_block_cells(self, state: GameStateHex) -> set[tuple[int, int]]:
        """Cases où l'adversaire gagne en 1 coup"""
        rep = cast(BoardHex, state.get_rep())
        opp = self._find_opponent(state)
        opp_type = "B" if self.piece_type == "R" else "R"
        threats = set()
        for pos in rep.get_empty():
            scores = state.compute_scores((pos, opp_type, opp.get_id()))
            if scores.get(opp.get_id(), 0) == 1:
                threats.add(pos)
        return threats

    def _edge_reduce_for_player(self, state: GameStateHex, piece_type: str) -> set[tuple[int,int]]:
        """P5: Détection template 3→2 (réduction de bord)"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        
        edge_moves = set()
        d0 = self._connection_distance(state, piece_type)

        # Helper: la composante contenant (i,j) touche-t-elle un bord donné ?
        def component_touches_side(start: tuple[int,int], side: str) -> bool:
            sp = env.get(start)
            try:
                if sp is None or sp.get_type() != piece_type:
                    return False
            except Exception:
                return False
            stack = [start]
            visited: set[tuple[int,int]] = {start}
            while stack:
                ci, cj = stack.pop()
                if side == "TOP" and ci == 0:
                    return True
                if side == "BOTTOM" and ci == n - 1:
                    return True
                if side == "LEFT" and cj == 0:
                    return True
                if side == "RIGHT" and cj == n - 1:
                    return True
                for n_type, (ni, nj) in state.get_neighbours(ci, cj).values():
                    if n_type == piece_type and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        stack.append((ni, nj))
            return False
        
        # Chercher pierres en rang 3 (tous bords)
        for (i, j), p in env.items():
            try:
                if p.get_type() != piece_type:
                    continue
            except:
                continue

            candidates: list[tuple[int,int]] = []

            if piece_type == "R":
                # Haut: i==2 -> pousser vers i==1, rejeter si déjà connecté au TOP
                if i == 2:
                    if component_touches_side((i, j), "TOP"):
                        continue
                    candidates.extend([(1, j), (1, j-1), (1, j+1)])
                # Bas: i==n-3 -> pousser vers i==n-2, rejeter si déjà connecté au BOTTOM
                if i == n - 3:
                    if component_touches_side((i, j), "BOTTOM"):
                        continue
                    candidates.extend([(n - 2, j), (n - 2, j-1), (n - 2, j+1)])
            else:  # Bleu
                # Gauche: j==2 -> pousser vers j==1, rejeter si déjà connecté au LEFT
                if j == 2:
                    if component_touches_side((i, j), "LEFT"):
                        continue
                    candidates.extend([(i, 1), (i-1, 1), (i+1, 1)])
                # Droite: j==n-3 -> pousser vers j==n-2, rejeter si déjà connecté au RIGHT
                if j == n - 3:
                    if component_touches_side((i, j), "RIGHT"):
                        continue
                    candidates.extend([(i, n - 2), (i-1, n - 2), (i+1, n - 2)])

            if not candidates:
                continue
            
            for cand in candidates:
                ci, cj = cand
                if not (0 <= ci < n and 0 <= cj < n):
                    continue
                if env.get(cand) is not None:
                    continue
                
                # Vérifier que ça ne dégrade pas la distance de connexion
                action = LightAction({"piece": piece_type, "position": cand})
                child = cast(GameStateHex, state.apply_action(action))
                d1 = self._connection_distance(child, piece_type)
                if d1 <= d0:
                    edge_moves.add(cand)
        
        return edge_moves

    def _edge_reduce_rank2_to_rank1_for_player(self, state: GameStateHex, piece_type: str) -> set[tuple[int,int]]:
        """P5: Réduction de bord 2→1.
        Ne déclenche que si la composante alliée contenant le pion de rang 2 ne touche pas déjà
        le bord cible concerné par la poussée (TOP/BOTTOM pour R, LEFT/RIGHT pour B).
        Couvre les deux côtés: i==1 et i==n-2 (R), j==1 et j==n-2 (B).
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]

        edge_moves: set[tuple[int,int]] = set()
        d0 = self._connection_distance(state, piece_type)

        # Helper local: BFS sur la composante alliée et test côté
        def component_touches_side(start: tuple[int,int], side: str) -> bool:
            sp = env.get(start)
            try:
                if sp is None or sp.get_type() != piece_type:
                    return False
            except Exception:
                return False
            stack = [start]
            visited: set[tuple[int,int]] = {start}
            while stack:
                ci, cj = stack.pop()
                if side == "TOP" and ci == 0:
                    return True
                if side == "BOTTOM" and ci == n - 1:
                    return True
                if side == "LEFT" and cj == 0:
                    return True
                if side == "RIGHT" and cj == n - 1:
                    return True
                for n_type, (ni, nj) in state.get_neighbours(ci, cj).values():
                    if n_type == piece_type and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        stack.append((ni, nj))
            return False

        def component_nodes(start: tuple[int,int]) -> set[tuple[int,int]]:
            sp = env.get(start)
            try:
                if sp is None or sp.get_type() != piece_type:
                    return set()
            except Exception:
                return set()
            stack = [start]
            visited: set[tuple[int,int]] = {start}
            while stack:
                ci, cj = stack.pop()
                for n_type, (ni, nj) in state.get_neighbours(ci, cj).values():
                    if n_type == piece_type and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        stack.append((ni, nj))
            return visited

        def adjacent_to_component(cand: tuple[int,int], comp: set[tuple[int,int]]) -> bool:
            for _, (ni, nj) in state.get_neighbours(cand[0], cand[1]).values():
                if (ni, nj) in comp:
                    return True
            return False

        for (i, j), p in env.items():
            try:
                if p.get_type() != piece_type:
                    continue
            except Exception:
                continue

            cases: list[tuple[str, list[tuple[int,int]]]] = []

            if piece_type == "R":
                # Poussée vers TOP si en rang 2 haut: voisins de (i,j) sur i==0 sont (0,j) et (0,j+1)
                if i == 1:
                    cases.append(("TOP", [(0, j), (0, j+1)]))
                # Poussée vers BOTTOM si en rang 2 bas: voisins sur i==n-1 sont (n-1,j) et (n-1,j-1)
                if i == n - 2:
                    cases.append(("BOTTOM", [(n-1, j), (n-1, j-1)]))
            else:  # B
                # Poussée vers LEFT si en rang 2 gauche: voisins sur j==0 sont (i,0) et (i+1,0)
                if j == 1:
                    cases.append(("LEFT", [(i, 0), (i+1, 0)]))
                # Poussée vers RIGHT si en rang 2 droite: voisins sur j==n-1 sont (i,n-1) et (i-1,n-1)
                if j == n - 2:
                    cases.append(("RIGHT", [(i, n-1), (i-1, n-1)]))

            if not cases:
                continue

            for side, candidates in cases:
                # Ne rien proposer si la composante touche déjà ce bord
                if component_touches_side((i, j), side):
                    continue
                # Composante alliée de référence (rang 2)
                comp = component_nodes((i, j))

                for cand in candidates:
                    ci, cj = cand
                    if not (0 <= ci < n and 0 <= cj < n):
                        continue
                    if env.get(cand) is not None:
                        continue
                    # Le coup doit être collé à la composante rang 2
                    if not adjacent_to_component(cand, comp):
                        continue

                    # Garder si ça n'aggrave pas (idéalement réduit) la distance de connexion
                    action = LightAction({"piece": piece_type, "position": cand})
                    child = cast(GameStateHex, state.apply_action(action))
                    d1 = self._connection_distance(child, piece_type)
                    if d1 <= d0:
                        edge_moves.add(cand)

        return edge_moves

    # (supprimé) _edge_double_threat_from_last_move: remplacé prochainement par le module Blocking
 
    # ========== ZIGGURAT TEMPLATE (P7+) ==========
 
    def _ziggurat_specs_raw(self, state: GameStateHex, anchor: tuple[int,int], piece_type: str, anchor_side: str = "L") -> Optional[dict]:
        """
        Gabarit brut (sans vérif des limites). Renvoie orientation, 9 positions (4-3-2) et escapes (A,B)
        ou None si l'ancre n'est pas sur un rang admissible.
        Règle bord: on décale le triangle pour rester in-bounds (évite j±3/i±3 hors plateau).
        anchor_side: "L" (ancre à gauche du rang 2) ou "R" (ancre à droite du rang 2).
        """
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
        i, j = anchor

        orientation: Optional[str] = None
        A: Optional[tuple[int,int]] = None
        B: Optional[tuple[int,int]] = None

        if piece_type == "B":
            if j == 2:
                # C-left (bord gauche)
                # Ancre à gauche (par défaut): row2 = [(i, j),(i+1, j)] avec ancre = (i, j)
                # Ancre à droite: décaler i de -1 pour que (i, j) soit la 2e case de row2
                if anchor_side == "R":
                    i = i - 1
                orientation = "B:C-left"
                row2 = [(i, j), (i+1, j)]
                row3 = [(i, j-1), (i+1, j-1), (i+2, j-1)]
                row4 = [(i, j-2), (i+1, j-2), (i+2, j-2), (i+3, j-2)]
                # Escapes = extrémités du rang 3
                A, B = (i, j-1), (i+2, j-1)
            elif j == n - 3:
                # L-right (bord droit)
                # Ancre à gauche (par défaut): row2 = [(i-1, j),(i, j)] avec ancre = (i, j)
                # Ancre à droite: décaler i de +1 pour que (i, j) soit la 1ère case de row2
                if anchor_side == "R":
                    i = i + 1
                orientation = "B:L-right"
                row2 = [(i-1, j), (i, j)]
                row3 = [(i-2, j+1), (i-1, j+1), (i, j+1)]
                row4 = [(i-3, j+2), (i-2, j+2), (i-1, j+2), (i, j+2)]
                A, B = (i, j+1), (i-2, j+1)
            else:
                return None
        else:
            if i == 2:
                # 3-up (bord haut)
                # Ancre à gauche (par défaut): row2 = [(i, j),(i, j+1)] avec ancre = (i, j)
                # Ancre à droite: décaler j de -1 pour que (i, j) soit la 2e case de row2
                if anchor_side == "R":
                    j = j - 1
                orientation = "R:3-up"
                row2 = [(i, j), (i, j+1)]
                row3 = [(i-1, j), (i-1, j+1), (i-1, j+2)]
                row4 = [(i-2, j), (i-2, j+1), (i-2, j+2), (i-2, j+3)]
                A, B = (i-1, j), (i-1, j+2)
            elif i == n - 3:
                # 12-down (bord bas)
                # Ancre à gauche (par défaut): row2 = [(i, j-1),(i, j)] avec ancre = (i, j)
                # Ancre à droite: décaler j de +1 pour que (i, j) soit la 1ère case de row2
                if anchor_side == "R":
                    j = j + 1
                orientation = "R:12-down"
                row2 = [(i, j-1), (i, j)]
                row3 = [(i+1, j-2), (i+1, j-1), (i+1, j)]
                row4 = [(i+2, j-3), (i+2, j-2), (i+2, j-1), (i+2, j)]
                A, B = (i+1, j), (i+1, j-2)
            else:
                return None

        rows = row2 + row3 + row4
        return {"orientation": orientation, "carrier": rows, "escapes": (A, B), "anchor_side": anchor_side}

    def _ziggurat_specs(self, state: GameStateHex, anchor: tuple[int,int], piece_type: str, anchor_side: str = "L") -> Optional[dict]:
        """
        Spécifie un ziggurat valide (triangle 4-3-2 = 9 cases uniques) en vérifiant in-bounds.
        """
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]

        raw = self._ziggurat_specs_raw(state, anchor, piece_type, anchor_side)
        if not raw:
            return None

        rows = raw["carrier"]
        if len(rows) != 9 or len(set(rows)) != 9:
            return None

        for (ci, cj) in rows:
            if not (0 <= ci < n and 0 <= cj < n):
                return None

        return raw
 
    def _detect_ziggurat(self, state: GameStateHex, anchor: tuple[int,int], piece_type: str) -> bool:
        """
        Détection stricte d’un ziggurat selon specs (carrier entièrement vide).
        Logs détaillés pour diagnostic.
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        i, j = anchor
 
        p = env.get(anchor)
        if not p or p.get_type() != piece_type:
            # ziggurat debug removed
            return False
 
        spec = self._ziggurat_specs(state, anchor, piece_type)
        if not spec:
            # ziggurat debug removed
            return False
 
        carrier = spec["carrier"]
        A, B = spec["escapes"]
        # ziggurat debug removed
        if len(carrier) != 9 or len(set(carrier)) != 9:
            print(f"[MyPlayer][zig-detect] anchor={self._pos_to_an(anchor, n)} fail=carrier_len_or_duplicates")
            return False
 
        ok = True
        for (ci, cj) in carrier:
            inb = (0 <= ci < n and 0 <= cj < n)
            occ = env.get((ci, cj))
            occ_t = None
            try:
                occ_t = occ.get_type() if occ is not None else None
            except Exception:
                occ_t = None
            # ziggurat debug removed
            if not inb:
                ok = False
            elif (ci, cj) != anchor and occ_t is not None:
                ok = False
        if not ok:
            # ziggurat debug removed
            return False
 
        # ziggurat debug removed
        return True
 
    def _get_ziggurat_escape_cells(self, anchor: tuple[int,int], piece_type: str) -> tuple[tuple[int,int], tuple[int,int]]:
        """
        Retourne les deux cases A (bas-gauche) et B (bas-droite) pour un ziggurat.
        Ces cases sont les sorties du template.
        """
        i, j = anchor
        
        if piece_type == "R":
            # A = bas-gauche, B = bas-droite (relativement au mouvement vers le bas)
            A = (i+1, j-1)
            B = (i+1, j)
        else:  # Bleu
            # A = bas-gauche, B = bas-droite (relativement au mouvement vers la droite)
            A = (i-1, j+1)
            B = (i, j+1)
        
        return A, B
 
    def _classify_ziggurat_intrusion(self, anchor: tuple[int,int], intruder: tuple[int,int], piece_type: str) -> str:
        """
        Détermine si l'intrusion est du côté gauche ou droit du ziggurat.
        """
        ai, aj = anchor
        ii, ij = intruder
        
        if piece_type == "R":
            # Pour Rouge (vertical), on compare les lignes
            if ii < ai:
                return "left"
            else:
                return "right"
        else:  # Bleu
            # Pour Bleu (horizontal), on compare les colonnes
            if ij < aj:
                return "left"
            else:
                return "right"
 
    def _classify_zig_with_orient(self, anchor: tuple[int,int], intruder: tuple[int,int], orient: str, anchor_side: str) -> str:
        ai, aj = anchor
        ii, ij = intruder
        # Offsets relatifs (di,dj) depuis l'ancre, pour ancre à gauche ("L") et à droite ("R")
        masks = {
            "B:L-right": {
                "L": {
                    "left": [(0, 1), (0, 2), (-1, 2)],
                    "right": [(-1, 0), (-1, 1), (-2, 1), (-2, 2), (-3, 2)],
                },
                "R": {
                    # ancre à droite (ex L12): gauche = L13,M12,M13,N12,N13; droite = M11,N11,N10
                    "left": [(+1, 0), (0, +1), (+1, +1), (0, +2), (+1, +2)],
                    "right": [(-1, +1), (-1, +2), (-2, +2)],
                },
            },
            "B:C-left": {
                "L": {
                    "left": [(0, -2), (0, -1), (1, -2)],
                    "right": [(1, 0), (1, -1), (2, -2), (2, -1), (3, -2)],
                },
                "R": {
                    # ancre à droite (ex C9): gauche = A8,B8,C8,A9,B9 ; droite = A11,B11,A12
                    "left": [(-1, -2), (-1, -1), (-1, 0), (0, -2), (0, -1)],
                    "right": [(+2, -2), (+2, -1), (+3, -2)],
                },
            },
            "R:12-down": {
                "L": {
                    "left": [(1, -1), (2, -2), (2, -1)],
                    "right": [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
                },
                "R": {
                    # ancre à droite (ex G12): gauche = F12,E13,F13,D14,E14 ; droite = G13,F14,G14
                    "left": [(0, -1), (+1, -2), (+1, -1), (+2, -3), (+2, -2)],
                    "right": [(+1, 0), (+2, -1), (+2, 0)],
                },
            },
            "R:3-up": {
                "L": {
                    "left": [(-2, 1), (-2, 2), (-1, 1)],
                    "right": [(-2, -1), (-2, 0), (-1, -1), (-1, 0), (0, -1)],
                },
                "R": {
                    # ancre à droite (ex G3): gauche = H3,H2,I2,I1,J1 ; droite = G1,H1,G2
                    "left": [(0, +1), (-1, +1), (-1, +2), (-2, +2), (-2, +3)],
                    "right": [(-2, 0), (-2, +1), (-1, 0)],
                },
            },
        }
        mo = masks.get(orient)
        if not mo:
            return "unknown"
        m = mo.get(anchor_side, {})
        di, dj = ii - ai, ij - aj
        if (di, dj) in m.get("left", []):
            return "left"
        if (di, dj) in m.get("right", []):
            return "right"
        return "unknown"

    def _ziggurat_intrusion_response(self, state: GameStateHex, piece_type: str, last_pos: Optional[tuple[int,int]]) -> Optional[tuple[int,int]]:
        """
        Réponse à une intrusion (toutes variantes). Tolère exactement 1 pierre adverse (last_pos) dans le carrier.
        Ajoute des raisons explicites pour 'no_specs':
         - not_allowed_anchor (rang invalide)
         - out_of_bounds=[...]
         - carrier_len=...
         - duplicates=[...]
         - not_empty=[...] (liste des cases qui bloquent, hors last_pos)
         - last_pos_not_in_carrier / last_pos_not_opp
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]

        if not last_pos:
            # ziggurat debug removed
            return None

        # ziggurat debug removed
        opp_type = "B" if piece_type == "R" else "R"
        p = env.get(last_pos)
        if not p or p.get_type() != opp_type:
            print("[MyPlayer][zig] skip: last_pos not opponent piece")
            return None

        # Construire la liste des ancres admissibles par rang
        anchors: list[tuple[int,int]] = []
        for (i, j), pc in env.items():
            try:
                if pc.get_type() != piece_type:
                    continue
            except Exception:
                continue
            if piece_type == "B" and (j == 2 or j == n - 3):
                anchors.append((i, j))
            elif piece_type == "R" and (i == 2 or i == n - 3):
                anchors.append((i, j))
        # ziggurat debug removed

        for anchor in anchors:
            anchor_an = self._pos_to_an(anchor, n)

            tried_any = False
            # Essayer les deux interprétations d'ancre: gauche puis droite
            for a_side in ("L", "R"):
                # Specs brutes (4-3-2) sans vérif bornes pour diagnostiquer précisément
                raw = self._ziggurat_specs_raw(state, anchor, piece_type, a_side)
                if not raw:
                    if a_side == "L":
                        print(f"[MyPlayer][zig] {anchor_an} no_specs (reason not_allowed_anchor)")
                    continue
                tried_any = True

                rows = raw["carrier"]
                A_raw, B_raw = raw["escapes"]

                # Vérifs structurelles
                if len(rows) != 9:
                    # ziggurat debug removed
                    continue
                if len(set(rows)) != 9:
                    # extraire les doublons
                    seen = set()
                    dups = []
                    for r in rows:
                        if r in seen and r not in dups:
                            dups.append(r)
                        seen.add(r)
                    dups_an = [self._pos_to_an(r, n) for r in dups if 0 <= r[0] < n and 0 <= r[1] < n]
                    # ziggurat debug removed
                    continue

                # Vérif bornes
                oob = [(ci, cj) for (ci, cj) in rows if not (0 <= ci < n and 0 <= cj < n)]
                if oob:
                    oob_an = [self._pos_to_an(c, n) if (0 <= c[0] < n and 0 <= c[1] < n) else str(c) for c in oob]
                    # ziggurat debug removed
                    continue

                # Vérif "état avant intrusion": tout vide sauf éventuellement last_pos (adversaire)
                blockers = []
                intruder_inside = False
                for (ci, cj) in rows:
                    # Ignorer l'ancre (elle est à nous et peut être occupée)
                    if (ci, cj) == anchor:
                        continue
                    occ = env.get((ci, cj))
                    occ_t = None
                    try:
                        occ_t = occ.get_type() if occ is not None else None
                    except Exception:
                        occ_t = None
                    if (ci, cj) == last_pos:
                        intruder_inside = (occ_t == opp_type)
                    elif occ_t is not None:
                        blockers.append((ci, cj))

                if blockers:
                    # ziggurat debug removed
                    continue
                if not intruder_inside:
                    # soit last_pos n'est pas dedans, soit ce n'est pas une pierre adverse
                    inside = last_pos in rows
                    reason = "last_pos_not_in_carrier" if not inside else "last_pos_not_opp"
                    # ziggurat debug removed
                    continue

                # Ici: ziggurat détecté avec intrusion
                spec = {"orientation": raw["orientation"], "carrier": rows, "escapes": (A_raw, B_raw), "anchor_side": a_side}
                A, B = spec["escapes"]
                # ziggurat debug removed

                a_free = env.get(A) is None
                b_free = env.get(B) is None
                # ziggurat debug removed

                choice: Optional[tuple[int,int]] = None
                orient = spec["orientation"]
                side = self._classify_zig_with_orient(anchor, last_pos, orient, a_side)

                # Règle générale: côté gauche => préférer B, côté droit => préférer A
                if side == "left":
                    pref = "B"
                elif side == "right":
                    pref = "A"
                else:
                    pref = None

                # ziggurat debug removed

                if pref == "A":
                    choice = A if a_free else (B if b_free else None)
                elif pref == "B":
                    choice = B if b_free else (A if a_free else None)
                else:
                    # Fallback: choisir l'escape la plus éloignée de l'intrus si les deux sont libres
                    if a_free and b_free:
                        li, lj = last_pos
                        dA = abs(li - A[0]) + abs(lj - A[1])
                        dB = abs(li - B[0]) + abs(lj - B[1])
                        choice = A if dA > dB else B
                    elif a_free:
                        choice = A
                    elif b_free:
                        choice = B

                if choice:
                    # ziggurat debug removed (return choice)
                    return choice

            if not tried_any:
                continue

        # ziggurat debug removed
        return None
 
    def _ziggurat_formation_bonus(self, state: GameStateHex, pos: tuple[int,int]) -> float:
        """
        Bonus si jouer pos crée une ancre de ziggurat valide (4 variantes).
        """
        i, j = pos
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
 
        # Pos doit être sur un rang d'ancre autorisé
        if self.piece_type == "B" and not (j == 2 or j == n - 3):
            return 0.0
        if self.piece_type == "R" and not (i == 2 or i == n - 3):
            return 0.0
 
        # Simuler le coup et valider via specs
        child = cast(GameStateHex, state.apply_action(LightAction({"piece": self.piece_type, "position": pos})))
        if not self._detect_ziggurat(child, pos, self.piece_type):
            return 0.0
 
        spec = self._ziggurat_specs(child, pos, self.piece_type)
        if not spec:
            return 0.0
 
        n_child = cast(BoardHex, child.get_rep()).get_dimensions()[0]
        A, B = spec["escapes"]
        return 0.20
 
    # ========== FIN ZIGGURAT ==========
 
    def _bridge_intrusion_replies(self, state: GameStateHex, piece_type: str, last_pos: Optional[tuple[int,int]]) -> list[tuple[tuple[int,int], tuple[int,int], tuple[int,int], tuple[int,int]]]:
        """P4: Détection réactive d'intrusion dans un bridge"""
        if not last_pos:
            return []
        
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        opp_type = "B" if piece_type == "R" else "R"
        
        intr = env.get(last_pos)
        if not intr or intr.get_type() != opp_type:
            return []
        
        li, lj = last_pos
        
        def neigh_coords(i, j):
            s = set()
            for n_type, (ni, nj) in state.get_neighbours(i, j).values():
                if n_type != "OUTSIDE":
                    s.add((ni, nj))
            return s
        
        def are_neighbors(a, b):
            for _, (ni, nj) in state.get_neighbours(a[0], a[1]).values():
                if (ni, nj) == b:
                    return True
            return False
        
        last_neigh = []
        for n_type, (ni, nj) in state.get_neighbours(li, lj).values():
            if n_type == "EMPTY":
                last_neigh.append((ni, nj))
        
        replies = []
        N_last = neigh_coords(li, lj)
        
        for ei, ej in last_neigh:
            N_e = neigh_coords(ei, ej)
            commons = N_last & N_e
            allies = []
            for ci, cj in commons:
                p = env.get((ci, cj))
                if p and p.get_type() == piece_type:
                    allies.append((ci, cj))
            if len(allies) != 2:
                continue
            a1, a2 = allies
            if are_neighbors(a1, a2):
                continue
            replies.append(((ei, ej), a1, a2, last_pos))
        
        return replies

    def _handle_bridge_replies(self, state: GameStateHex, replies: list, actions: list[LightAction]) -> Optional[LightAction]:
        """Gère les réponses aux intrusions de bridge"""
        d_now = self._connection_distance(state, self.piece_type)
        reply_positions = {rp for (rp, _, _, _) in replies}
        candidates = [a for a in actions if a.data.get("position") in reply_positions]
        
        best_bridge = None
        best_val = float("-inf")
        
        for a in candidates:
            child = cast(GameStateHex, state.apply_action(a))
            if self._connection_distance(child, self.piece_type) <= d_now:
                v = self._fast_static_eval(child)
                if v > best_val:
                    best_val = v
                    best_bridge = a
        
        if best_bridge:
            rep = cast(BoardHex, state.get_rep())
            n = rep.get_dimensions()[0]
            pos = best_bridge.data.get("position")
            if isinstance(pos, tuple):
                print(f"[MyPlayer] bridge_intrusion reply={self._pos_to_an(pos, n)}")
        
        return best_bridge

    def _choose_best_from_candidates(self, state: GameStateHex, candidates: list[LightAction]) -> Optional[LightAction]:
        """Choisit le meilleur coup parmi une liste de candidats"""
        best = None
        best_val = float("-inf")
        for a in candidates:
            child = cast(GameStateHex, state.apply_action(a))
            v = self._fast_static_eval(child)
            if v > best_val:
                best_val = v
                best = a
        return best

    # ========== ÉPINE DORSALE & BRIDGES (P3.2) ==========

    def _guiding_spine(self, state: GameStateHex, piece_type: str) -> list[tuple[int,int]]:
        """Calcule l'épine dorsale (plus court chemin bord→bord)"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        
        def cell_type(i, j):
            p = env.get((i, j))
            if p is None:
                return "EMPTY"
            try:
                return p.get_type()
            except:
                return "EMPTY"
        
        sources, goals = [], set()
        if piece_type == "R":
            for j in range(n):
                t = cell_type(0, j)
                if t != "B":
                    sources.append((0, j))
            for j in range(n):
                goals.add((n-1, j))
        else:
            for i in range(n):
                t = cell_type(i, 0)
                if t != "R":
                    sources.append((i, 0))
            for i in range(n):
                goals.add((i, n-1))
        
        INF = 10**9
        dist = [[INF]*n for _ in range(n)]
        par: list[list[Optional[tuple[int, int]]]] = [[None for _ in range(n)] for _ in range(n)]
        pq = []
        
        for i, j in sources:
            t = cell_type(i, j)
            if t == piece_type:
                d = 0
            elif t == "EMPTY":
                d = 1
            else:
                continue
            dist[i][j] = d
            heapq.heappush(pq, (d, (i, j)))
        
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > dist[i][j]:
                continue
            if (i, j) in goals:
                path = []
                cur = (i, j)
                while cur is not None:
                    path.append(cur)
                    pi, pj = cur
                    cur = par[pi][pj]
                path.reverse()
                return path
            
            for n_type, (ni, nj) in state.get_neighbours(i, j).values():
                if n_type == "OUTSIDE":
                    continue
                if n_type == piece_type:
                    nd = d
                elif n_type == "EMPTY":
                    nd = d + 1
                else:
                    nd = INF
                
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    par[ni][nj] = cast(tuple[int, int], (i, j))
                    heapq.heappush(pq, (nd, (ni, nj)))
        
        return []

    def _dist_to_spine(self, pos: tuple[int,int], spine: list[tuple[int,int]]) -> int:
        """Distance Manhattan approximative à l'épine"""
        if not spine:
            return 3
        pi, pj = pos
        best = 10**9
        for si, sj in spine:
            d = abs(pi - si) + abs(pj - sj)
            if d < best:
                best = d
            if best == 0:
                break
        return best

    def _bridge_creation_info(self, state: GameStateHex, piece_type: str, pos: tuple[int,int]) -> list[tuple[set, tuple[int,int], bool]]:
        """Détecte si jouer pos crée/complète un bridge"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        
        if env.get(pos) is not None:
            return []
        
        i, j = pos
        patterns = [
            # Seules paires non-voisines (véritables “bridges”)
            ((+1,+1), [(0,+1),(+1,0)]),      # diamant nord-est
            ((-1,-1), [(-1,0),(0,-1)]),      # diamant sud-ouest
            ((+2,-1), [(+1,0),(+1,-1)]),     # diamant sud-est
            ((-2,+1), [(-1,0),(-1,+1)]),     # diamant nord-ouest
        ]
        
        def inb(x, y):
            return 0 <= x < n and 0 <= y < n
        
        out = []
        for dx, dy in [p[0] for p in patterns]:
            bi, bj = i + dx, j + dy
            if not inb(bi, bj):
                continue
            pB = env.get((bi, bj))
            if not pB or pB.get_type() != piece_type:
                continue
            
            hrels = None
            for p, hr in patterns:
                if p == (dx, dy):
                    hrels = hr
                    break
            if not hrels:
                continue
            
            holes = []
            for hr in hrels:
                hi, hj = i + hr[0], j + hr[1]
                if inb(hi, hj):
                    holes.append((hi, hj))
            
            if len(holes) == 2:
                is_comp = False
                for h in holes:
                    q = env.get(h)
                    if q and q.get_type() == piece_type:
                        is_comp = True
                        break
                out.append((set(holes), (bi, bj), is_comp))
        
        return out

    def _log_bridge_decision(self, state: GameStateHex, pos: tuple[int,int], source: str) -> None:
        """Log léger si le coup pos forme un bridge (create/complete) selon _bridge_creation_info."""
        try:
            info = self._bridge_creation_info(state, self.piece_type, pos)
            if not info:
                return
            # Choisir une entrée “complete” en priorité, sinon la première
            entry = None
            for holes, ally, is_comp in info:
                if is_comp:
                    entry = (holes, ally, is_comp)
                    break
            if entry is None:
                entry = info[0]
            holes, ally, is_comp = entry
            rep = cast(BoardHex, state.get_rep())
            n = rep.get_dimensions()[0]
            holes_an = [self._pos_to_an(h, n) for h in holes]
            ally_an = self._pos_to_an(ally, n)
            pos_an = self._pos_to_an(pos, n)
            kind = "complete" if is_comp else "create"
            print(f"[MyPlayer] bridge posé kind={kind} pos={pos_an} ally={ally_an} holes={holes_an} source={source}")
        except Exception:
            pass

    def _bridge_guided_bonus(self, state: GameStateHex, pos: tuple[int,int], spine: list[tuple[int,int]]) -> float:
        """P3.2: Bonus épine + bridges + gradient"""
        if not (isinstance(pos, tuple) and len(pos) == 2):
            return 0.0
        
        # Proximité épine
        dsp = self._dist_to_spine(pos, spine)
        proximity = 0.12 * (1.0 / (1 + dsp))
        
        # Gradient distance
        d0 = self._connection_distance(state, self.piece_type)
        child = cast(GameStateHex, state.apply_action(LightAction({"piece": self.piece_type, "position": pos})))
        d1 = self._connection_distance(child, self.piece_type)
        gradient = 0.0
        if d1 < d0:
            gradient += 0.12
        elif dsp >= 2:
            gradient -= 0.06
        
        # Bridges
        info = self._bridge_creation_info(state, self.piece_type, pos)
        if info:
            BRIDGE_COMPLETE = 0.40
            BRIDGE_CREATE = 0.25
            best = 0.0
            best_entry = None
            for holes, ally, is_comp in info:
                near = min(self._dist_to_spine(h, spine) for h in holes) <= 1
                if near:
                    cand = BRIDGE_COMPLETE if is_comp else BRIDGE_CREATE
                    if cand > best:
                        best = cand
                        best_entry = (holes, ally, is_comp)
            if best_entry is not None and best > 0.0:
                # Log discret: _bridge_creation_info a contribué au score
                rep = cast(BoardHex, state.get_rep())
                n = rep.get_dimensions()[0]
                holes_an = [self._pos_to_an(h, n) for h in best_entry[0]]
                ally_an = self._pos_to_an(best_entry[1], n)
                pos_an = self._pos_to_an(pos, n)
                kind = "complete" if best_entry[2] else "create"
            return best + proximity + gradient
        
        return proximity + gradient

    # ========== ÉVALUATION & HEURISTIQUES ==========

    def _utility(self, state: GameState) -> int:
        """Utilité terminale: +1 si je gagne, -1 sinon"""
        scores = state.get_scores()
        my_id = self.get_id()
        if scores.get(my_id, 0) == 1:
            return 1
        for pid, sc in scores.items():
            if pid != my_id and sc == 1:
                return -1
        return 0

    def _evaluate(self, state: GameStateHex) -> float:
        """P2: Évaluation H1 (base + distance)"""
        if state.is_done():
            return float(self._utility(state))
        
        base = self._fast_static_eval(state)
        
        my_type = self.piece_type
        opp_type = "B" if my_type == "R" else "R"
        
        my_dist = self._connection_distance(state, my_type)
        opp_dist = self._connection_distance(state, opp_type)
        
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
        dist_term = (opp_dist - my_dist) / max(1, n)
        
        return float(0.35 * base + 0.65 * dist_term)

    def _fast_static_eval(self, state: GameStateHex) -> float:
        """Évaluation rapide: pions, ancrage, présence, centre, cluster"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        
        my_type = self.piece_type
        opp_type = "B" if my_type == "R" else "R"
        
        # Pions
        our, opp = 0, 0
        for p in env.values():
            t = p.get_type()
            if t == my_type:
                our += 1
            elif t == opp_type:
                opp += 1
        piece_diff = (our - opp) / max(1, our + opp)
        
        # Centre
        c_my = self._center_score(state, my_type)
        c_opp = self._center_score(state, opp_type)
        center_term = (c_opp - c_my) / max(1, n)
        # Option B: renforcer le centre en ouverture (log debug)
        total_stones = len(env)
        if total_stones <= 10:
            f_open = 3.0
        elif total_stones >= 20:
            f_open = 1.0
        else:
            f_open = 3.0 - ((total_stones - 10) / 5.0)
        center_term *= float(f_open)

        
        # Ancrage
        a_my = self._edge_anchor_score(state, my_type)
        a_opp = self._edge_anchor_score(state, opp_type)
        anchor_term = (a_opp - a_my) / max(1, n)
        
        # Cluster
        cl_my = self._cluster_score(state, my_type)
        cl_opp = self._cluster_score(state, opp_type)
        cluster_term = (cl_opp - cl_my) / 6.0
        
        # Présence bords
        presence_term = self._edge_presence_score(state, my_type) - self._edge_presence_score(state, opp_type)
        
        val = 0.25 * piece_diff + 0.35 * anchor_term + 0.20 * presence_term + 0.15 * center_term + 0.05 * cluster_term

        # Petit terme bridge (doux): +0.05 créer, +0.01 compléter, seulement si trous libres et pas de must-block
        try:
            if not self._must_block_cells(state):
                has_create, has_complete = self._bridge_eval_flags(state, my_type)
                if has_create:
                    val += 0.10
                if has_complete:
                    val += 0.01
        except Exception:
            pass

        return max(-0.9, min(0.9, val))

    def _root_order_score(self, state: GameStateHex) -> float:
        """Score pour ordonnancement racine: base + distance + ancrage"""
        base = self._fast_static_eval(state)
        
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
        my_type = self.piece_type
        opp_type = "B" if my_type == "R" else "R"
        
        d_my = self._connection_distance(state, my_type)
        d_opp = self._connection_distance(state, opp_type)
        dist_term = (d_opp - d_my) / max(1, n)
        
        a_my = self._edge_anchor_score(state, my_type)
        a_opp = self._edge_anchor_score(state, opp_type)
        anchor_term = (a_opp - a_my) / max(1, n)
        
        val = 0.35 * base + 0.45 * dist_term + 0.20 * anchor_term
        return max(-0.9, min(0.9, val))

    # ========== HELPERS MÉTRIQUES ==========

    def _find_opponent(self, state: GameStateHex):
        for p in state.players:
            if p.get_id() != self.get_id():
                return p
        return state.players[0]

    def _edge_anchor_score(self, state: GameStateHex, piece_type: str) -> int:
        """Distance min aux deux bords cibles"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        
        if piece_type == "R":
            top, bot = n, n
            for (i, j), p in env.items():
                if p.get_type() == piece_type:
                    if i < top:
                        top = i
                    d_bot = (n - 1) - i
                    if d_bot < bot:
                        bot = d_bot
            return top + bot if (top != n or bot != n) else n
        else:
            left, right = n, n
            for (i, j), p in env.items():
                if p.get_type() == piece_type:
                    if j < left:
                        left = j
                    d_right = (n - 1) - j
                    if d_right < right:
                        right = d_right
            return left + right if (left != n or right != n) else n

    def _cluster_score(self, state: GameStateHex, piece_type: str) -> float:
        """Nombre moyen de voisins alliés par pièce"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        tot, count = 0, 0
        for (i, j), p in env.items():
            if p.get_type() != piece_type:
                continue
            same = 0
            for n_type, _ in state.get_neighbours(i, j).values():
                if n_type == piece_type:
                    same += 1
            tot += same
            count += 1
        return tot / float(count) if count else 0.0

    def _edge_presence_score(self, state: GameStateHex, piece_type: str) -> float:
        """1.0 si présent sur les 2 bords, 0.5 si 1 seul, 0.0 sinon"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        
        has_top = has_bottom = has_left = has_right = False
        for (i, j), p in env.items():
            if p.get_type() != piece_type:
                continue
            if piece_type == "R":
                if i == 0:
                    has_top = True
                if i == n - 1:
                    has_bottom = True
            else:
                if j == 0:
                    has_left = True
                if j == n - 1:
                    has_right = True
        
        if piece_type == "R":
            return 1.0 if (has_top and has_bottom) else (0.5 if (has_top or has_bottom) else 0.0)
        else:
            return 1.0 if (has_left and has_right) else (0.5 if (has_left or has_right) else 0.0)

    def _bridge_holes(self, state: GameStateHex, piece_type: str) -> set[tuple[int,int]]:
        """Détecte tous les trous de bridges"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        holes = set()
        opp_type = "B" if piece_type == "R" else "R"
        n = rep.get_dimensions()[0]
        
        def inb(i, j):
            return 0 <= i < n and 0 <= j < n
        
        def consider_pair(h1, h2):
            if not (inb(*h1) and inb(*h2)):
                return
            s1 = env.get(h1)
            s2 = env.get(h2)
            if s1 is None and s2 is None:
                holes.add(h1)
                holes.add(h2)
                return
            if s1 is None and s2 is not None:
                try:
                    if s2.get_type() == opp_type:
                        holes.add(h1)
                except:
                    pass
            elif s2 is None and s1 is not None:
                try:
                    if s1.get_type() == opp_type:
                        holes.add(h2)
                except:
                    pass
        
        for (i, j), p in env.items():
            if p.get_type() != piece_type:
                continue
            
            # 4 diagonales + 2 verticaux
            pairs = [
                ((i+1, j+1), [(i, j+1), (i+1, j)]),
                ((i-1, j-1), [(i-1, j), (i, j-1)]),
                ((i-1, j+1), [(i-1, j), (i, j+1)]),
                ((i+1, j-1), [(i, j-1), (i+1, j)]),
                ((i+2, j-1), [(i+1, j), (i+1, j-1)]),
                ((i-2, j+1), [(i-1, j), (i-1, j+1)]),
            ]
            
            for (pi, pj), hole_pair in pairs:
                if inb(pi, pj):
                    pp = env.get((pi, pj))
                    if pp and pp.get_type() == piece_type:
                        consider_pair(hole_pair[0], hole_pair[1])
        
        return holes

    def _bridge_eval_flags(self, state: GameStateHex, piece_type: str) -> tuple[bool, bool]:
        """Détecte la présence d'au moins un motif bridge côté 'piece_type'.
        Retourne (has_create, has_complete):
          - create: 2 trous vides pour un couple allié non-voisin (diamant)
          - complete: 1 trou vide et 1 trou déjà allié
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]

        def inb(i: int, j: int) -> bool:
            return 0 <= i < n and 0 <= j < n

        has_create = False
        has_complete = False

        # Offsets B - A pour les 4 diamants et trous relatifs à A
        patterns: list[tuple[int,int,list[tuple[int,int]]]] = [
            (+1, +1, [(0, +1), (+1, 0)]),
            (-1, -1, [(-1, 0), (0, -1)]),
            (+2, -1, [(+1, 0), (+1, -1)]),
            (-2, +1, [(-1, 0), (-1, +1)]),
        ]

        for (ai, aj), pA in env.items():
            try:
                if pA.get_type() != piece_type:
                    continue
            except Exception:
                continue
            for dx, dy, holes_rel in patterns:
                bi, bj = ai + dx, aj + dy
                if not inb(bi, bj):
                    continue
                pB = env.get((bi, bj))
                if not pB or pB.get_type() != piece_type:
                    continue
                holes: list[tuple[int,int]] = []
                for hr in holes_rel:
                    hi, hj = ai + hr[0], aj + hr[1]
                    if inb(hi, hj):
                        holes.append((hi, hj))
                if len(holes) != 2:
                    continue
                occ_types: list[Optional[str]] = []
                for h in holes:
                    q = env.get(h)
                    try:
                        occ_types.append(q.get_type() if q is not None else None)
                    except Exception:
                        occ_types.append(None)
                empties = [t is None for t in occ_types]
                allies = [t == piece_type for t in occ_types]
                if empties[0] and empties[1]:
                    has_create = True
                elif (empties[0] and allies[1]) or (empties[1] and allies[0]):
                    has_complete = True
                if has_create and has_complete:
                    return True, True

        return has_create, has_complete

    def _classify_bridge_move(self, state: GameStateHex, pos: tuple[int,int]) -> Optional[str]:
        """Classe 'pos' sur l'état courant: 'create' si deux trous vides, 'complete' si 1 vide et 1 allié; sinon None."""
        try:
            info = self._bridge_creation_info(state, self.piece_type, pos)
            if not info:
                return None
            rep = cast(BoardHex, state.get_rep())
            env = rep.get_env()
            for holes, ally, is_comp in info:
                hs = list(holes)
                occ_types: list[Optional[str]] = []
                for h in hs:
                    q = env.get(h)
                    try:
                        occ_types.append(q.get_type() if q is not None else None)
                    except Exception:
                        occ_types.append(None)
                empties = [t is None for t in occ_types]
                allies = [t == self.piece_type for t in occ_types]
                if empties[0] and empties[1]:
                    return "create"
                if (allies[0] and empties[1]) or (allies[1] and empties[0]):
                    return "complete"
            return None
        except Exception:
            return None

    def _friendly_neighbors_count(self, state: GameStateHex, pos: tuple[int,int], piece_type: str) -> int:
        """Nombre de voisins alliés"""
        count = 0
        for n_type, _ in state.get_neighbours(pos[0], pos[1]).values():
            if n_type == piece_type:
                count += 1
        return count

    def _border_reduce_candidates_from_actions(self, state: GameStateHex, piece_type: str, d0: Optional[int], actions: list[LightAction]) -> set[tuple[int,int]]:
        """Candidats de réduction bord (anneau 2)"""
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
        if d0 is None:
            d0 = self._connection_distance(state, piece_type)
        
        def in_ring2(pos):
            i, j = pos
            if piece_type == "R":
                return i == 1 or i == n - 2
            else:
                return j == 1 or j == n - 2
        
        cands = set()
        for a in actions:
            pos = a.data.get("position")
            if not (isinstance(pos, tuple) and len(pos) == 2 and in_ring2(pos)):
                continue
            child = cast(GameStateHex, state.apply_action(a))
            d1 = self._connection_distance(child, piece_type)
            if d1 <= d0 - 1:
                cands.add(pos)
        return cands

    def _connection_distance(self, state: GameStateHex, piece_type: str) -> int:
        """Distance de connexion bord-à-bord par Dijkstra 0-1"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        
        def cell_type(i, j):
            p = env.get((i, j))
            if p is None:
                return "EMPTY"
            try:
                return p.get_type()
            except:
                return "EMPTY"
        
        dist = [[float("inf")] * n for _ in range(n)]
        pq = []
        
        if piece_type == "R":
            goals = {(n - 1, j) for j in range(n)}
            for j in range(n):
                t = cell_type(0, j)
                if t == "EMPTY":
                    dist[0][j] = 1
                elif t == piece_type:
                    dist[0][j] = 0
                else:
                    continue
                heapq.heappush(pq, (dist[0][j], (0, j)))
        else:
            goals = {(i, n - 1) for i in range(n)}
            for i in range(n):
                t = cell_type(i, 0)
                if t == "EMPTY":
                    dist[i][0] = 1
                elif t == piece_type:
                    dist[i][0] = 0
                else:
                    continue
                heapq.heappush(pq, (dist[i][0], (i, 0)))
        
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > dist[i][j]:
                continue
            if (i, j) in goals:
                return int(d)
            
            for n_type, (ni, nj) in state.get_neighbours(i, j).values():
                if n_type == "OUTSIDE":
                    continue
                if n_type == "EMPTY":
                    nd = d + 1
                elif n_type == piece_type:
                    nd = d
                else:
                    continue
                
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    heapq.heappush(pq, (nd, (ni, nj)))
        
        return n * 2

    def _center_score(self, state: GameStateHex, piece_type: str) -> float:
        """Score moyen de distance au centre (plus petit = mieux)"""
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        mid = (n - 1) / 2.0
        
        s, c = 0.0, 0
        for (i, j), p in env.items():
            try:
                t = p.get_type()
            except:
                t = None
            if t == piece_type:
                s += abs(i - mid) + abs(j - mid)
                c += 1
        
        return s / c if c else float(n)

    def _pos_to_an(self, pos: tuple[int, int], n: int) -> str:
        """Convertit une position (i,j) en notation algébrique (ex: A1)"""
        i, j = pos
        col = chr(ord('A') + j)
        return f"{col}{i + 1}"