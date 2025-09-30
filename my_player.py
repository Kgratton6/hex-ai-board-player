from typing import Optional
import random

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from game_state_hex import GameStateHex


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

    def compute_action(self, current_state: GameState, remaining_time: int = 1_000_000_000, **kwargs) -> Action:  # type: ignore[override]
        """
        Sélectionne une action via Minimax alpha-bêta avec profondeur limitée et heuristique basique.
        """
        state: GameState = current_state
        actions = list(state.get_possible_light_actions())
        if not actions:
            raise RuntimeError("Aucune action légale disponible.")

        best_action: Optional[LightAction] = None
        best_value = float("-inf")
        alpha, beta = float("-inf"), float("inf")

        def quick_eval(a: LightAction) -> float:
            return self._evaluate(state.apply_action(a))

        actions.sort(key=quick_eval, reverse=True)

        for action in actions:
            child = state.apply_action(action)
            value = self._alphabeta(child, depth=self.max_depth - 1, alpha=alpha, beta=beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        return best_action if best_action is not None else actions[0]

    def _alphabeta(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
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
                child = state.apply_action(action)
                value = max(value, self._alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for action in state.get_possible_light_actions():
                child = state.apply_action(action)
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

    def _evaluate(self, state: GameState) -> float:
        """
        Heuristique basique:
        - "simple": (mes pions - pions adverses) / max(1, mes+pions adverses), borné dans [-0.9, 0.9]
        - "random": bruit uniforme dans [-0.1, 0.1]
        """
        if state.is_done():
            return float(self._utility(state))

        if self._heuristic == "random":
            return random.uniform(-0.1, 0.1)

        env = state.get_rep().get_env()
        our_type = self.piece_type
        opp_type = "B" if our_type == "R" else "R"
        our = 0
        opp = 0
        for piece in env.values():
            t = piece.get_type()
            if t == our_type:
                our += 1
            elif t == opp_type:
                opp += 1

        denom = max(1, our + opp)
        val = (our - opp) / denom
        if val > 0.9:
            val = 0.9
        if val < -0.9:
            val = -0.9
        return float(val)
