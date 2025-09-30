import copy
import json
from typing import Dict, Generator, List, Optional, Set, Tuple

from board_hex import BoardHex
from player_hex import PlayerHex
from seahorse.game.game_layout.board import Piece
from seahorse.game.game_state import GameState
from seahorse.game.heavy_action import HeavyAction
from seahorse.game.light_action import LightAction
from seahorse.player.player import Player
from seahorse.utils.serializer import Serializable
from seahorse.player.proxies import RemotePlayerProxy, LocalPlayerProxy, InteractivePlayerProxy

class GameStateHex(GameState):
    """
    A class representing the state of an Hex game.

    Attributes:
        score (list[float]): Scores of the state for each player.
        next_player (Player): Next player to play.
        players (list[Player]): List of players.
        rep (Representation): Representation of the game.
    """
    def __init__(self, scores: Dict, next_player: Player, players: List[Player], rep: BoardHex, step: int,  *args, **kwargs) -> None:
        super().__init__(scores, next_player, players, rep)
        self.max_step = rep.get_dimensions()[0] * rep.get_dimensions()[1] #+ 1 #+1 for the swap
        self.step = step

    def get_step(self) -> int:
        """
        Return the current step of the game.

        Returns:
            int: The current step of the game.
        """
        return self.step

    def is_done(self) -> bool:
        """
        Check if the game is finished.

        Returns:
            bool: True if the game is finished, False otherwise.
        """
        if self.scores[self.players[0].id] == 1 or self.scores[self.players[1].id] == 1:
            return True
        return False

    def get_neighbours(self, i: int, j: int) -> Dict[str,Tuple[str,Tuple[int,int]]]:
        return self.get_rep().get_neighbours(i, j)

    def in_board(self, index) -> bool:
        """
        Check if a given index is within the game board.

        Args:
            index: The index to check.

        Returns:
            bool: True if the index is within the game board, False otherwise.
        """
        if index[0] < 0 or index[0] >= self.get_rep().get_dimensions()[0] or index[1] < 0 or index[1] >= self.get_rep().get_dimensions()[1]:
            return False
        return True
    
    def get_player_id(self, pid) -> Player:
        """
        Get the player with the given ID.

        Args:
            pid: The ID of the player.

        Returns:
            Player: The player with the given ID.
        """
        for player in self.players:
            if player.get_id() == pid:
                return player
    
    def generate_possible_heavy_actions(self) -> Generator[HeavyAction, None, None]:
        """
        Generate possible actions.

        Returns:
            Generator[HeavyAction]: Generator of possible heavy actions.
        """
        for position in self.rep.get_empty():
            piece_type = self.next_player.get_piece_type()
            current_rep = self.get_rep()
            b = current_rep.get_env()
            d = current_rep.get_dimensions()
            copy_b = copy.copy(b)
            copy_b[position] = Piece(piece_type=piece_type, owner=self.next_player)
            new_board = BoardHex(env=copy_b, dim=d)
            play_info = (position, piece_type, self.next_player.get_id())
            new_state =  GameStateHex(
                self.compute_scores(play_info=play_info),
                self.compute_next_player(),
                self.players,
                new_board,
                step=self.step + 1,
            )
            yield HeavyAction(self, new_state)

                        

    def generate_possible_light_actions(self) -> Generator[LightAction, None, None]:
        """
        Generate possible light actions for the current game state.

        Returns:
            Generator[LightAction]: Generator of possible light actions.

        """
        for position in self.rep.get_empty():
            yield LightAction({"piece": self.next_player.get_piece_type(), "position": position})

    def apply_action(self, action: LightAction) -> GameState:
        """
        Apply an action to the game state.

        Args:
            action (LightAction): The action to apply.

        Returns:
            GameState: The new game state.
        """
        if not isinstance(action, LightAction):
            raise ValueError("The action must be a LightAction.")
        
        piece_type, position = action.data["piece"], action.data["position"]
        current_rep = self.get_rep()
        
        if current_rep.get_env().get(position) is not None:
            # If the position is already occupied, we return the current state without changes
            # This is a fallback to ensure the game state remains valid
            return GameStateHex(
                self.scores,
                self.next_player,
                self.players,
                current_rep,
                step=self.step,
            )
        
        b = current_rep.get_env()
        d = current_rep.get_dimensions()
        copy_b = copy.copy(b)
        copy_b[position] = Piece(piece_type=piece_type, owner=self.next_player)
        new_board = BoardHex(env=copy_b, dim=d)
        play_info = (position, piece_type, self.next_player.get_id())
        return GameStateHex(
            self.compute_scores(play_info=play_info),
            self.compute_next_player(),
            self.players,
            new_board,
            step=self.step + 1,
        )
    
    def convert_heavy_action_to_light_action(self, heavy_action: HeavyAction) -> LightAction:
        """
        Generate a light action from a heavy action.

        Args:
            heavy_action (HeavyAction): The heavy action to convert.

        Returns:
            LightAction: The generated light action.
        """
        if not isinstance(heavy_action, HeavyAction):
            raise ValueError("The heavy action must be an instance of HeavyAction.")
        
        current_state = heavy_action.get_current_game_state()
        next_state = heavy_action.get_next_game_state()
        
        dim = current_state.get_rep().get_dimensions()[0]
        for i in range(dim):
            for j in range(dim):
                if current_state.get_rep().get_env().get((i, j)) != next_state.get_rep().get_env().get((i, j)):
                    piece_type = next_state.get_rep().get_env().get((i, j)).get_type()
                    return LightAction({"piece": piece_type, "position": (i, j)})
        raise ValueError("No light action found in the heavy action.")

    def convert_gui_data_to_action_data(self, gui_data: dict) -> dict:
        """
        Convert GUI data to action data.

        Args:
            gui_data (dict): The GUI data to convert.

        Returns:
            dict: The converted action data.
        """
        return {"piece": gui_data["piece"], "position": tuple(gui_data["position"])}
    
    def compute_scores(self, play_info: tuple) -> Dict[int, float]:
        """
        Compute the score of each player in a list.

        Args:
            id_add (int): The ID of the player to add the score for.

        Returns:
            dict[int, float]: A dictionary with player ID as the key and score as the value.
        """
        player1, player2 = self.players[0].id, self.players[1].id
        pos, piece_type, id_player = play_info
        if self.get_rep().get_env().get(pos) is not None and self.step == 1:
            return {player1: 0.0, player2: 0.0}
        self.get_rep().get_env()[pos] = Piece(piece_type=piece_type, owner=self.get_player_id(id_player))
        dim = self.get_rep().get_dimensions()[0]
        visited = set()
        def dfs_bot(i,j):
            if (i,j) in visited:
                return False
            visited.add((i,j))
            if i == dim-1:
                return True
            for k, v in self.get_neighbours(i,j).items():
                t, (ni, nj) = v
                if t == piece_type and (ni,nj) not in visited:
                    if dfs_bot(ni,nj):
                        return True
            return False
        
        def dfs_right(i,j):
            if (i,j) in visited:
                return False
            visited.add((i,j))
            if j == dim-1:
                return True
            for k, v in self.get_neighbours(i,j).items():
                t, (ni, nj) = v
                if t == piece_type and (ni,nj) not in visited:
                    if dfs_right(ni,nj):
                        return True
            return False

        if id_player == player1:
            for j in range(dim):
                piece = self.get_rep().get_env().get((0,j), -1)
                if piece != -1 and piece.get_type() == piece_type:
                    if dfs_bot(0,j):
                        self.get_rep().get_env().pop(pos)
                        return {player1: 1.0, player2: 0.0}
        else:
            for i in range(dim):
                piece = self.get_rep().get_env().get((i,0), -1)
                if piece != -1 and piece.get_type() == piece_type:
                    if dfs_right(i,0):
                        self.get_rep().get_env().pop(pos)
                        return {player1: 0.0, player2: 1.0}
        self.get_rep().get_env().pop(pos)
        return  {player1: 0, player2: 0}

    def __str__(self) -> str:
        if not self.is_done():
            return super().__str__()
        return "The game is finished!"

    def to_json(self) -> str:
        data = { i:j for i,j in self.__dict__.items() if i!="_possible_light_actions" and i!="_possible_heavy_actions"}
        data["step"] = self.step
        return data
    
    @classmethod
    def from_json(cls,data:str,*,next_player:Optional[PlayerHex]=None) -> Serializable:
        d = json.loads(data)
        return cls(**{**d,"scores":{int(k):v for k,v in d["scores"].items()},"players":[PlayerHex.from_json(json.dumps(x)) if not isinstance(x,str) else next_player for x in d["players"]],"next_player":next_player,"rep":BoardHex.from_json(json.dumps(d["rep"])),"step": d.get("step", 0)})

def copy_player_with_new_piece_type(player, new_piece_type):
    """This method, used when swapping players, allows to copy a player with a new piece type, while keeping the other attributes intact."""
    cls = type(player)

    if isinstance(player, RemotePlayerProxy):
        orig_mimics = player.mimics
        new_mimics = copy_player_with_new_piece_type(orig_mimics, new_piece_type)
        new_proxy = cls(lambda *args, **kwargs: new_mimics)
        new_proxy.mimics = new_mimics
        new_proxy.sid = player.sid
        new_proxy.sio = player.sio
        new_proxy.connected = player.connected
        return new_proxy

    if isinstance(player, InteractivePlayerProxy):
        new_wrapped = copy_player_with_new_piece_type(player.wrapped_player, new_piece_type)
        new_proxy = cls(new_wrapped, gui_path=player.path)
        new_proxy.shared_sid = player.shared_sid
        new_proxy.sid = player.sid
        new_proxy.sio = player.sio
        new_proxy.connected = player.connected
        return new_proxy

    if isinstance(player, LocalPlayerProxy):
        new_wrapped = copy_player_with_new_piece_type(player.wrapped_player, new_piece_type)
        new_proxy = cls(new_wrapped)
        new_proxy.sio = player.sio
        new_proxy.connected = player.connected
        return new_proxy

    # Base case (Not a proxy)
    name = getattr(player, 'name', 'bob')
    new_player = cls(new_piece_type, name)
    # Copy other fields by reference
    exclude = {'piece_type', 'name'}
    for attr, val in player.__dict__.items():
        if attr not in exclude:
            setattr(new_player, attr, val)
    return new_player



def get_constructor_args(player):
    return (player.piece_type, getattr(player, 'name', 'bob'))

