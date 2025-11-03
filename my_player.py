from typing import Optional, cast, Any
import random
import time

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from game_state_hex import GameStateHex
from board_hex import BoardHex


class MyPlayer(PlayerHex):
    """
    Joueur Hex avec Minimax + élagage alpha-bêta.
    Heuristique simple (différence de pions) ou aléatoire, au choix.
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer", max_depth: int = 3, heuristic: str = "simple"):
        """
        Args:
            piece_type (str): "R" ou "B"
            name (str): nom du joueur
            max_depth (int): profondeur de recherche (>=1)
            heuristic (str): "simple" (diff de pions) ou "random"
        """
        super().__init__(piece_type, name)
        self.max_depth = max(1, int(max_depth))
        self._heuristic = heuristic if heuristic in ("simple", "random") else "simple"
        # Stats internes (non sérialisées si préfixées par '_')
        self._nodes_visited = 0
        self._depth_reached = 0
        self._last_stats = {}
        self._deadline: float = float("inf")
        # Snapshot des cases occupées (pour déduire le dernier coup adverse)
        self._prev_positions: set[tuple[int, int]] = set()

    def compute_action(self, current_state: GameState, remaining_time: int = 1_000_000_000, **kwargs) -> Action:  # type: ignore[override]
        """
        Sélectionne une action via Minimax alpha-bêta avec profondeur limitée et heuristique basique.
        """
        # Compteurs et horloge
        start = time.perf_counter()
        self._nodes_visited = 0
        self._depth_reached = 0

        # Budget temps par coup (soft): 2% du temps restant, borné [0.2s, 0.75s]
        try:
            rt = float(remaining_time)
        except Exception:
            rt = 900.0
        per_move_budget = min(0.75, max(0.2, 0.02 * rt))
        self._deadline = start + per_move_budget

        state: GameState = current_state
        st_hex = cast(GameStateHex, state)
        actions = list(state.get_possible_light_actions())
        if not actions:
            raise RuntimeError("Aucune action légale disponible.")
        # Déduire la dernière position jouée (adverse) via comparaison d’état
        rep_board = cast(BoardHex, st_hex.get_rep())
        env = rep_board.get_env()
        curr_pos_set = set(env.keys())
        last_pos: Optional[tuple[int, int]] = None
        try:
            prev = getattr(self, "_prev_positions", set())
            added_list = list(curr_pos_set - prev)
            opp_type_now = "B" if self.piece_type == "R" else "R"
            # Accès robuste au type
            def _ptype_at(pos: tuple[int,int]):
                p = env.get(pos)
                if p is None:
                    return None
                try:
                    return p.get_type()
                except Exception:
                    return getattr(p, "piece_type", None)
            opp_added = [pos for pos in added_list if _ptype_at(pos) == opp_type_now]

            if len(opp_added) == 1:
                last_pos = opp_added[0]
            elif len(opp_added) > 1:
                # Ambiguïté rare: garder la première (mieux que None; on loggue le détail)
                last_pos = opp_added[0]
            else:
                # Aucun ajout adverse détecté; laisser last_pos=None mais tracer le delta complet
                last_pos = None

            # Traces détaillées pour diagnostic
            try:
                n_dim = rep_board.get_dimensions()[0]
                def fmt_list(L):
                    return "[" + ",".join(self._pos_to_an(p, n_dim) for p in L) + "]"
                print(f"[MyPlayer] added={fmt_list(added_list)} opp_added={fmt_list(opp_added)}")
            except Exception:
                pass
        except Exception:
            last_pos = None
        # Log du dernier coup adverse déduit
        try:
            n_dim = rep_board.get_dimensions()[0]
            if last_pos is not None:
                print(f"[MyPlayer] last_opp={self._pos_to_an(last_pos, n_dim)}")
            else:
                print("[MyPlayer] last_opp=?")
        except Exception:
            pass

        # Must-block: si l'adversaire gagne en 1 coup, bloquer immédiatement
        threats = self._must_block_cells(st_hex)
        if threats:
            candidates = [a for a in actions if a.data.get("position") in threats]
            if candidates:
                best_block: Optional[LightAction] = None
                best_val = float("-inf")
                for a in candidates:
                    child_b = cast(GameStateHex, st_hex.apply_action(a))
                    v = self._fast_static_eval(child_b)
                    if v > best_val:
                        best_val = v
                        best_block = a
                if best_block is not None:
                    try:
                        print(f"[MyPlayer] must_block size={len(candidates)}")
                    except Exception:
                        pass
                    # Mettre à jour le snapshot en y ajoutant notre coup choisi
                    try:
                        pos_played = best_block.data.get("position")
                        if isinstance(pos_played, tuple) and len(pos_played) == 2:
                            self._prev_positions = set(curr_pos_set)
                            self._prev_positions.add(pos_played)
                    except Exception:
                        pass
                    return best_block

        # Bridge réactif uniquement: si l'adversaire vient d'occuper un des deux trous d'un de nos ponts,
        # jouer l'autre trou immédiatement (ne pas remplir nos ponts proactivement).
        replies = self._bridge_intrusion_replies(st_hex, self.piece_type, last_pos)
        if replies:
            d_now = self._connection_distance(st_hex, self.piece_type)
            reply_positions = {rp for (rp, _, _, _) in replies}
            bridge_candidates = [a for a in actions if a.data.get("position") in reply_positions]
            best_bridge: Optional[LightAction] = None
            best_info: Optional[tuple[tuple[int,int], tuple[int,int], tuple[int,int], tuple[int,int]]] = None
            best_val = float("-inf")
            for a in bridge_candidates:
                child_b = cast(GameStateHex, st_hex.apply_action(a))
                # Par sécurité, ne joue que si cela ne dégrade pas la distance de connexion
                if self._connection_distance(child_b, self.piece_type) <= d_now:
                    v = self._fast_static_eval(child_b)
                    if v > best_val:
                        best_val = v
                        best_bridge = a
                        # associer l'info correspondante
                        pos = a.data.get("position")
                        for info in replies:
                            if info[0] == pos:
                                best_info = info
                                break
            if best_bridge is not None:
                # Log explicite de la détection
                try:
                    if best_info is not None:
                        rp, a1, a2, intr = best_info
                        n_dim = rep_board.get_dimensions()[0]
                        rp_s   = self._pos_to_an(rp, n_dim)
                        a1_s   = self._pos_to_an(a1, n_dim)
                        a2_s   = self._pos_to_an(a2, n_dim)
                        intr_s = self._pos_to_an(intr, n_dim)
                        print(f"[MyPlayer] bridge_intrusion allies={a1_s},{a2_s} intruder={intr_s} reply={rp_s}")
                    else:
                        print("[MyPlayer] bridge_intrusion reply chosen")
                except Exception:
                    pass
                # Mettre à jour le snapshot en y ajoutant notre coup choisi
                try:
                    pos_played = best_bridge.data.get("position")
                    if isinstance(pos_played, tuple) and len(pos_played) == 2:
                        self._prev_positions = set(curr_pos_set)
                        self._prev_positions.add(pos_played)
                except Exception:
                    pass
                return best_bridge

        best_action: Optional[LightAction] = None
        best_value = float("-inf")
        alpha, beta = float("-inf"), float("inf")

        # Tri en deux étages pour limiter le coût (distance seulement sur top_k)
        base_scored = []
        for a in actions:
            child_tmp = cast(GameStateHex, st_hex.apply_action(a))
            base_scored.append((a, self._fast_static_eval(child_tmp)))
        # Tri initial par éval rapide
        base_scored.sort(key=lambda t: t[1], reverse=True)
        # Pré-calcul: trous de pont de l'adversaire (éviter d'y jouer sans nécessité)
        opp_type = "B" if self.piece_type == "R" else "R"
        opp_bridge_holes_all = self._bridge_holes(st_hex, opp_type)

        # Pré-calcul: candidats de réduction bord (rangée/colonne 2) qui réduisent la distance de connexion
        curr_dist_conn = self._connection_distance(st_hex, self.piece_type)
        border_reduce_candidates = self._border_reduce_candidates_from_actions(st_hex, self.piece_type, curr_dist_conn, actions)
        try:
            n_dim = rep_board.get_dimensions()[0]
            br_list = ",".join(self._pos_to_an(p, n_dim) for p in border_reduce_candidates)
            print(f"[MyPlayer] border_candidates=[{br_list}]")
        except Exception:
            pass

        # Affinage sur un sous-ensemble (top_k) avec distance; borne temps ~ min(0.25 s, 1/3 du budget)
        refined = []
        # Debug ordonnancement: (posAN, base, final, used_stage2, adj_bonus, bh_penalty)
        order_debug: list[tuple[str, float, float, bool, float, bool]] = []
        n_dim = rep_board.get_dimensions()[0]
        start_order = time.perf_counter()
        top_k = min(16, max(6, len(base_scored) // 5))
        order_time_cap = min(0.25, max(0.08, (self._deadline - start) / 3.0))
        for idx, (a, base_val) in enumerate(base_scored):
            child_for_penalty: Optional[GameStateHex] = None
            pos = a.data.get("position")
            used_stage2 = False
            adj_bonus = 0.0
            if idx < top_k and (time.perf_counter() - start_order) < order_time_cap:
                child_top = cast(GameStateHex, st_hex.apply_action(a))
                score = self._root_order_score(child_top)
                child_for_penalty = child_top
                used_stage2 = True
                # Petit bonus d'adjacence locale (évite l'isolement complet)
                if isinstance(pos, tuple) and len(pos) == 2:
                    adj = self._friendly_neighbors_count(st_hex, cast(tuple[int,int], pos), self.piece_type)
                    adj_bonus = 0.08 * (adj / 6.0)
                    score += adj_bonus
            else:
                score = base_val
            # Micro-biais de flanc: si le dernier coup adverse est sur/près d'un bord,
            # privilégier l'anneau 2 du MÊME flanc (colonne 1 ou n-2 pour Bleu, ligne 1 ou n-2 pour Rouge)
            if isinstance(pos, tuple) and len(pos) == 2 and last_pos is not None:
                n_side = n_dim
                if self.piece_type == "B":
                    if last_pos[1] <= 1 and pos[1] == 1:
                        score += 0.06
                    elif last_pos[1] >= n_side - 2 and pos[1] == n_side - 2:
                        score += 0.06
                else:
                    if last_pos[0] <= 1 and pos[0] == 1:
                        score += 0.06
                    elif last_pos[0] >= n_side - 2 and pos[0] == n_side - 2:
                        score += 0.06
            # Bonus léger si coup de réduction bord détecté
            if isinstance(pos, tuple) and len(pos) == 2 and pos in border_reduce_candidates:
                score += 0.12
            # Pénalité légère si on joue dans un trou de pont adverse (sauf blocage ou gain immédiat)
            bh_penalty = False
            if isinstance(pos, tuple) and len(pos) == 2 and pos in opp_bridge_holes_all and (pos not in threats):
                if child_for_penalty is None:
                    child_for_penalty = cast(GameStateHex, st_hex.apply_action(a))
                if self._utility(child_for_penalty) != 1:
                    score -= 0.12
                    bh_penalty = True
            refined.append((a, score))
            # Enregistrer debug pour ce coup
            try:
                pos_s = self._pos_to_an(pos, n_dim) if isinstance(pos, tuple) and len(pos) == 2 else "?"
                order_debug.append((pos_s, float(base_val), float(score), used_stage2, float(adj_bonus), bh_penalty))
            except Exception:
                pass

        spent_ms = (time.perf_counter() - start_order) * 1000.0
        refined.sort(key=lambda t: t[1], reverse=True)
        actions = [a for (a, _) in refined]

        # Résumés d’ordonnancement
        try:
            print(f"[MyPlayer] order n={len(base_scored)} top_k={top_k} cap_ms={order_time_cap*1000:.1f} spent_ms={spent_ms:.1f}")
            dbg_map = {d[0]: d for d in order_debug}
            top_list = []
            for a, sc in refined[:5]:
                p = a.data.get("position")
                key = self._pos_to_an(p, n_dim) if isinstance(p, tuple) and len(p) == 2 else "?"
                d = dbg_map.get(key)
                flags = ""
                if d:
                    _, _, _, used_r2, adj_b, pen = d
                    if used_r2: flags += "*r2"
                    if adj_b > 0.0: flags += "+adj"
                    if pen: flags += "-bh"
                # Marquer bonus réduction bord
                if isinstance(p, tuple) and len(p) == 2 and p in border_reduce_candidates:
                    flags += "+br"
                # Marquer biais de flanc si applicable
                if isinstance(p, tuple) and len(p) == 2 and last_pos is not None:
                    if self.piece_type == "B":
                        if (last_pos[1] <= 1 and p[1] == 1) or (last_pos[1] >= n_dim - 2 and p[1] == n_dim - 2):
                            flags += "+fl"
                    else:
                        if (last_pos[0] <= 1 and p[0] == 1) or (last_pos[0] >= n_dim - 2 and p[0] == n_dim - 2):
                            flags += "+fl"
                top_list.append(f"{key}:{sc:.3f}{flags}")
            print(f"[MyPlayer] order_top5={','.join(top_list)}")
        except Exception:
            pass

        for action in actions:
            child = cast(GameStateHex, st_hex.apply_action(action))
            value = self._alphabeta(child, depth=self.max_depth - 1, alpha=alpha, beta=beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._last_stats = {
            "nodes": self._nodes_visited,
            "elapsed_ms": elapsed_ms,
            "depth_reached": self._depth_reached,
        }
        try:
            print(f"[MyPlayer] nodes={self._last_stats['nodes']} time_ms={self._last_stats['elapsed_ms']:.1f} depth={self._last_stats['depth_reached']}")
        except Exception:
            pass
        # Choix final
        chosen = best_action if best_action is not None else actions[0]
        # Log si le coup choisi est une réduction bord
        try:
            pos_ch = chosen.data.get("position")
            if isinstance(pos_ch, tuple) and len(pos_ch) == 2:
                if pos_ch in border_reduce_candidates:
                    n_dim = rep_board.get_dimensions()[0]
                    print(f"[MyPlayer] border_reduce={self._pos_to_an(pos_ch, n_dim)}")
        except Exception:
            pass
        # Mémoriser l’état APRÈS notre coup pour déduire exactement le prochain last_opp
        try:
            pos_played = chosen.data.get("position")
            if isinstance(pos_played, tuple) and len(pos_played) == 2:
                snap = set(curr_pos_set)
                snap.add(pos_played)
                self._prev_positions = snap
            else:
                self._prev_positions = curr_pos_set
        except Exception:
            self._prev_positions = curr_pos_set
        return chosen

    def _alphabeta(self, state: GameStateHex, depth: int, alpha: float, beta: float) -> float:
        # Garde temps: si deadline dépassée, retourne une éval rapide (coupe soft)
        if time.perf_counter() > self._deadline:
            return self._fast_static_eval(state)

        self._nodes_visited += 1
        # Mettre à jour la profondeur atteinte (à partir de la racine)
        try:
            reached = self.max_depth - depth
            if reached > self._depth_reached:
                self._depth_reached = reached
        except Exception:
            pass
        # Terminal exact
        if state.is_done():
            return float(self._utility(state))

        # Coupe: renvoie heuristique
        if depth <= 0:
            return float(self._evaluate(state))

        my_turn = (state.next_player.get_id() == self.get_id())

        if my_turn:
            value = float("-inf")
            for action in state.get_possible_light_actions():
                child = cast(GameStateHex, state.apply_action(action))
                value = max(value, self._alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for action in state.get_possible_light_actions():
                child = cast(GameStateHex, state.apply_action(action))
                value = min(value, self._alphabeta(child, depth - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _utility(self, state: GameState) -> int:
        """
        +1 si je gagne, -1 si l'adversaire gagne, 0 sinon (sécurité).
        """
        scores = state.get_scores()
        my_id = self.get_id()
        if scores.get(my_id, 0) == 1:
            return 1
        for pid, sc in scores.items():
            if pid != my_id and sc == 1:
                return -1
        return 0

    def _evaluate(self, state: GameStateHex) -> float:
        """
        Phase 2 — Evaluation principale combinant:
        - base rapide (_fast_static_eval): pions, ancrage, présence, centre, etc.
        - distance de connexion (opp_dist - my_dist) normalisée par n
        """
        # 1) terminal
        if state.is_done():
            return float(self._utility(state))

        # 2) mode aléatoire (inchangé)
        if self._heuristic == "random":
            return random.uniform(-0.1, 0.1)

        # 3) base actuelle
        base = self._fast_static_eval(state)

        # 4) distances de connexion
        my_type = self.piece_type
        opp_type = "B" if my_type == "R" else "R"

        my_dist = self._connection_distance(state, my_type)
        opp_dist = self._connection_distance(state, opp_type)

        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]

        # plus c'est grand, mieux c'est pour nous (ils sont plus loin que nous)
        dist_term = (opp_dist - my_dist) / max(1, n)

        # 5) combinaison avec poids significatif pour la distance
        return float(0.55 * base + 0.45 * dist_term)

    def _fast_static_eval(self, state: GameStateHex) -> float:
        """
        Heuristique rapide:
        val = 0.8 * piece_diff_norm + 0.2 * center_term
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]

        my_type = self.piece_type
        opp_type = "B" if my_type == "R" else "R"

        # Différence de pièces (normalisée)
        our = 0
        opp = 0
        for p in env.values():
            t = p.get_type()
            if t == my_type:
                our += 1
            elif t == opp_type:
                opp += 1
        denom = max(1, our + opp)
        piece_diff = (our - opp) / denom

        # Centralité relative
        c_my = self._center_score(state, my_type)
        c_opp = self._center_score(state, opp_type)
        center_term = (c_opp - c_my) / max(1, n)

        # Ancrage bords (plus proche des deux bords cibles = meilleur)
        a_my = self._edge_anchor_score(state, my_type)
        a_opp = self._edge_anchor_score(state, opp_type)
        anchor_term = (a_opp - a_my) / max(1, n)

        # Décongestion: préférer être moins aggloméré que l'adversaire
        cl_my = self._cluster_score(state, my_type)   # [0,6] voisins moyens
        cl_opp = self._cluster_score(state, opp_type)
        # positif si nous sommes moins agglomérés
        cluster_term = (cl_opp - cl_my) / 6.0

        # Présence sur les deux bords cibles (0, 0.5, 1.0)
        presence_term = self._edge_presence_score(state, my_type) - self._edge_presence_score(state, opp_type)

        # pondérations ajustées: renforcer l'ancrage et l'accès aux bords, atténuer la dispersion
        val = 0.42 * piece_diff + 0.33 * anchor_term + 0.19 * presence_term + 0.05 * center_term + 0.01 * cluster_term
        if val > 0.9:
            val = 0.9
        if val < -0.9:
            val = -0.9
        return float(val)

    def _root_order_score(self, state: GameStateHex) -> float:
        """
        Score plus coûteux pour l'ORDONNANCEMENT AU NIVEAU RACINE UNIQUEMENT:
        combine l'éval rapide + distance (Dijkstra 0-1) + ancrage bords.
        """
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

        val = 0.5 * base + 0.3 * dist_term + 0.2 * anchor_term
        if val > 0.9:
            val = 0.9
        if val < -0.9:
            val = -0.9
        return float(val)

    # ---------- Helpers défensifs / ancrage / dispersion ----------

    def _find_opponent(self, state: GameStateHex):
        for p in state.players:
            if p.get_id() != self.get_id():
                return p
        return state.players[0]

    def _must_block_cells(self, state: GameStateHex) -> set[tuple[int, int]]:
        """
        Retourne les cases où, si l'adversaire joue, il gagne immédiatement.
        """
        rep = cast(BoardHex, state.get_rep())
        opp = self._find_opponent(state)
        opp_type = "B" if self.piece_type == "R" else "R"
        threats: set[tuple[int, int]] = set()
        for pos in rep.get_empty():
            scores = state.compute_scores((pos, opp_type, opp.get_id()))
            if scores.get(opp.get_id(), 0) == 1:
                threats.add(pos)
        return threats

    def _edge_anchor_score(self, state: GameStateHex, piece_type: str) -> int:
        """
        Mesure d'ancrage aux bords cibles: somme des distances min aux deux bords.
        Plus petit = meilleur.
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        if piece_type == "R":
            # Bords haut (i=0) et bas (i=n-1)
            top = n
            bot = n
            for (i, j), p in env.items():
                if p.get_type() == piece_type:
                    if i < top:
                        top = i
                    d_bot = (n - 1) - i
                    if d_bot < bot:
                        bot = d_bot
            return top + bot if (top != n or bot != n) else n
        else:
            # Bords gauche (j=0) et droit (j=n-1)
            left = n
            right = n
            for (i, j), p in env.items():
                if p.get_type() == piece_type:
                    if j < left:
                        left = j
                    d_right = (n - 1) - j
                    if d_right < right:
                        right = d_right
            return left + right if (left != n or right != n) else n

    def _cluster_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Nombre moyen de voisins de même couleur par pièce (6 max). Plus petit = plus dispersé.
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        tot_neighbors = 0
        count = 0
        for (i, j), p in env.items():
            if p.get_type() != piece_type:
                continue
            same = 0
            for n_type, _ in state.get_neighbours(i, j).values():
                if n_type == piece_type:
                    same += 1
            tot_neighbors += same
            count += 1
        if count == 0:
            return 0.0
        return tot_neighbors / float(count)

    def _edge_presence_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Retourne 1.0 si au moins une pierre touche chacun des deux bords cibles,
        0.5 si un seul bord est touché, 0.0 sinon.
        - Rouge: bords haut (i=0) et bas (i=n-1)
        - Bleu:  bords gauche (j=0) et droit (j=n-1)
        """
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

    def _bridge_holes(self, state: GameStateHex, piece_type: str) -> set[tuple[int, int]]:
        """
        Détecte les trous (cases vides) de bridges (ponts) entre deux pierres alliées.
        Motifs couverts:
          - 4 diagonales: (+1,+1), (-1,-1), (-1,+1), (+1,-1)
          - 2 « verticaux » (sens axial): (+2,-1) et (-2,+1)
        Cas gérés:
          - les deux trous vides  -> renvoie les deux cases
          - un trou occupé par l'adversaire et l'autre vide -> renvoie la case vide (complétion urgente)
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        holes: set[tuple[int, int]] = set()
        opp_type = "B" if piece_type == "R" else "R"

        # Accès rapide au type d'une case
        def t(i: int, j: int):
            p = env.get((i, j))
            return None if p is None else p.get_type()

        n = rep.get_dimensions()[0]
        def inb(i: int, j: int) -> bool:
            return 0 <= i < n and 0 <= j < n

        def consider_pair(h1: tuple[int, int], h2: tuple[int, int]) -> None:
            if not (inb(*h1) and inb(*h2)):
                return
            s1 = env.get(h1)
            s2 = env.get(h2)
            if s1 is None and s2 is None:
                holes.add(h1); holes.add(h2)
                return
            # si un seul est vide et l'autre occupé par l'adversaire -> compléter le pont
            if s1 is None and s2 is not None:
                try:
                    if s2.get_type() == opp_type:
                        holes.add(h1)
                except Exception:
                    pass
            elif s2 is None and s1 is not None:
                try:
                    if s1.get_type() == opp_type:
                        holes.add(h2)
                except Exception:
                    pass

        for (i, j), p in env.items():
            if p.get_type() != piece_type:
                continue
            # diagonale (i+1, j+1): trous (i, j+1) et (i+1, j)
            pi, pj = i + 1, j + 1
            if inb(pi, pj) and t(pi, pj) == piece_type:
                consider_pair((i, j + 1), (i + 1, j))
            # diagonale (i-1, j-1): trous (i-1, j) et (i, j-1)
            pi, pj = i - 1, j - 1
            if inb(pi, pj) and t(pi, pj) == piece_type:
                consider_pair((i - 1, j), (i, j - 1))
            # diagonale (i-1, j+1): trous (i-1, j) et (i, j+1)
            pi, pj = i - 1, j + 1
            if inb(pi, pj) and t(pi, pj) == piece_type:
                consider_pair((i - 1, j), (i, j + 1))
            # diagonale (i+1, j-1): trous (i, j-1) et (i+1, j)
            pi, pj = i + 1, j - 1
            if inb(pi, pj) and t(pi, pj) == piece_type:
                consider_pair((i, j - 1), (i + 1, j))
            # « vertical » (i+2, j-1): trous (i+1, j) et (i+1, j-1)
            pi, pj = i + 2, j - 1
            if inb(pi, pj) and t(pi, pj) == piece_type:
                consider_pair((i + 1, j), (i + 1, j - 1))
            # « vertical » (i-2, j+1): trous (i-1, j) et (i-1, j+1)
            pi, pj = i - 2, j + 1
            if inb(pi, pj) and t(pi, pj) == piece_type:
                consider_pair((i - 1, j), (i - 1, j + 1))
        return holes

    def _bridge_intrusion_replies(
        self,
        state: GameStateHex,
        piece_type: str,
        last_pos: Optional[tuple[int, int]] = None
    ) -> list[tuple[tuple[int,int], tuple[int,int], tuple[int,int], tuple[int,int]]]:
        """
        Détecte de manière STRICTEMENT réactive une intrusion adverse dans un trou de pont:
        - On exige que last_pos soit connu ET soit effectivement une pierre adverse.
        - On considère chaque voisin vide E de last_pos. Si les deux voisins COMMUNS de (last_pos, E)
          sont occupés par deux alliés NON-VOISINS entre eux, alors (last_pos, E) sont les deux trous d’un pont.
          La réponse est de jouer E.
        - Renvoie une liste [(reply_pos, ally1, ally2, intruder_pos)].
        Cette méthode couvre automatiquement les 6 orientations (4 diagonales + 2 « verticales »).
        """
        if last_pos is None:
            return []
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        opp_type = "B" if piece_type == "R" else "R"
        li, lj = last_pos

        # Vérifier que last_pos contient bien une pierre adverse
        intr = env.get((li, lj))
        if intr is None or intr.get_type() != opp_type:
            return []

        def neigh_coords(i: int, j: int) -> set[tuple[int,int]]:
            s: set[tuple[int,int]] = set()
            for n_type, (ni, nj) in state.get_neighbours(i, j).values():
                if n_type != "OUTSIDE":
                    s.add((ni, nj))
            return s

        def are_neighbors(a: tuple[int,int], b: tuple[int,int]) -> bool:
            ai, aj = a
            for _, (ni, nj) in state.get_neighbours(ai, aj).values():
                if (ni, nj) == b:
                    return True
            return False

        last_neigh = []
        for n_type, (ni, nj) in state.get_neighbours(li, lj).values():
            if n_type == "EMPTY":
                last_neigh.append((ni, nj))

        replies: list[tuple[tuple[int,int], tuple[int,int], tuple[int,int], tuple[int,int]]] = []
        N_last = neigh_coords(li, lj)

        for ei, ej in last_neigh:
            N_e = neigh_coords(ei, ej)
            commons = N_last & N_e
            # Sélectionner exactement deux voisins communs alliés
            allies = []
            for ci, cj in commons:
                p = env.get((ci, cj))
                if p is not None and p.get_type() == piece_type:
                    allies.append((ci, cj))
            if len(allies) != 2:
                continue
            a1, a2 = allies[0], allies[1]
            # Écarter les faux positifs (les deux alliés collés l’un à l’autre)
            if are_neighbors(a1, a2):
                continue
            replies.append(((ei, ej), a1, a2, (li, lj)))

        return replies

    def _friendly_neighbors_count(self, state: GameStateHex, pos: tuple[int, int], piece_type: str) -> int:
        """
        Nombre de voisins alliés autour d'une position donnée (0..6) dans l'état fourni.
        """
        i, j = pos
        count = 0
        for n_type, _ in state.get_neighbours(i, j).values():
            if n_type == piece_type:
                count += 1
        return count

    def _pos_to_an(self, pos: tuple[int, int], n: int) -> str:
        i, j = pos
        col = chr(ord('A') + j)
        return f"{col}{i + 1}"

    def _border_reduce_candidates_from_actions(
        self,
        state: GameStateHex,
        piece_type: str,
        d0: Optional[int],
        actions: list[LightAction]
    ) -> set[tuple[int, int]]:
        """
        Détection simple des coups "réduction de bord" (3→2):
        - On ne considère que l'anneau 2 par rapport aux bords cibles:
          * Rouge (vertical): i == 1 ou i == n-2
          * Bleu  (horizontal): j == 1 ou j == n-2
        - Un coup est retenu s'il réduit la distance de connexion d'au moins 1.
        """
        rep = cast(BoardHex, state.get_rep())
        n = rep.get_dimensions()[0]
        if d0 is None:
            d0 = self._connection_distance(state, piece_type)

        def in_ring2(pos: tuple[int, int]) -> bool:
            i, j = pos
            if piece_type == "R":
                return i == 1 or i == n - 2
            else:
                return j == 1 or j == n - 2

        cands: set[tuple[int, int]] = set()
        for a in actions:
            pos = a.data.get("position")
            if not (isinstance(pos, tuple) and len(pos) == 2 and in_ring2(pos)):
                continue
            child = cast(GameStateHex, state.apply_action(a))
            d1 = self._connection_distance(child, piece_type)
            if d1 <= d0 - 1:
                cands.add(pos)
        return cands

 
    # ---------- Helpers H1 (distance/centre) ----------

    def _connection_distance(self, state: GameStateHex, piece_type: str) -> int:
        """
        Distance de connexion bord-à-bord par Dijkstra:
        - on traverse nos pierres à coût 0
        - les vides coûtent 1
        - les pierres adverses sont impassables
        Retourne une approximation de la longueur minimale.
        """
        import heapq

        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]

        def cell_type(i: int, j: int) -> str:
            p = env.get((i, j))
            if p is None:
                return "EMPTY"
            try:
                t = p.get_type()
            except Exception:
                t = getattr(p, "piece_type", None)
            return t if t is not None else "EMPTY"

        # Initialisation selon l’axe de connexion
        dist = [[float("inf")] * n for _ in range(n)]
        pq = []  # (d, (i,j))

        if piece_type == "R":
            # de la rangée 0 vers rangée n-1
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
            # de la colonne 0 vers colonne n-1
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

        # Dijkstra sur le graphe hex
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
                    # pièce adverse, impassable
                    continue
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    heapq.heappush(pq, (nd, (ni, nj)))

        # Non atteignable: renvoyer une valeur élevée mais bornée
        return n * 2

    def _center_score(self, state: GameStateHex, piece_type: str) -> float:
        """
        Score moyen de distance au centre pour les pierres d'un type (plus petit = mieux).
        Si aucune pierre, retourne n (pénalité neutre).
        """
        rep = cast(BoardHex, state.get_rep())
        env = rep.get_env()
        n = rep.get_dimensions()[0]
        mid = (n - 1) / 2.0

        s = 0.0
        c = 0
        for (i, j), p in env.items():
            try:
                t = p.get_type()
            except Exception:
                t = getattr(p, "piece_type", None)
            if t == piece_type:
                # distance "Manhattan" sur grille approximative
                s += abs(i - mid) + abs(j - mid)
                c += 1
        if c == 0:
            return float(n)
        return s / c
