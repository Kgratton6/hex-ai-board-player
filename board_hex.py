from __future__ import annotations
import json
from typing import Dict, List, Tuple, Generator
from colorama import Fore, Style
from seahorse.game.game_layout.board import Board, Piece
from seahorse.utils.serializer import Serializable

class BoardHex(Board):
    """
    A class representing an Hex board.

    Attributes:
        env (dict[Tuple[int], Piece]): The environment dictionary composed of pieces.
        dimension (int): The dimension of the board.
    """

    def __init__(self, env: dict[tuple[int], Piece], dim: int) -> None:
        super().__init__(env, dim)

    def __str__(self):
        grid_data = self.get_grid()
        board_string = ""
        for i in range(self.dimensions[0]):
            board_string += " " * i
            for j in range(self.dimensions[1]):
                cell = grid_data[i][j]
                if isinstance(cell, tuple):
                    char, color = cell
                    if color == 'R':
                        board_string += Fore.RED + char + Style.RESET_ALL + " "
                    elif color == 'B':
                        board_string += Fore.BLUE + char + Style.RESET_ALL + " "
                else:
                    board_string += cell + " "
            board_string += "\n"
        return board_string
    
    def get_neighbours(self, i:int ,j: int) -> Dict[str,Tuple[str|Piece,Tuple[int,int]]]:
        """ returns a dictionnary of the neighbours of the cell (i,j) with the following format:
        (neighbour_name: (neighbour_type, (i,j)))

        Args:
            i (int): line indice
            j (int): column indice

        Returns:
            Dict[str,Tuple[str,Tuple[int,int]]]: dictionnary of the neighbours of the cell (i,j)
        """
        neighbours = {"top_right":(i-1, j+1), "top_left":(i-1,j), "bot_left":(i+1, j-1), "bot_right":(i+1,j), "left":(i,j-1), "right":(i,j+1)}
        for k,v in neighbours.items():
            if v not in self.env.keys():
                if v[0] < 0 or v[1] < 0 or v[0] >= self.dimensions[0] or v[1] >= self.dimensions[1]:
                    neighbours[k] = ("OUTSIDE", neighbours[k])
                else:
                    neighbours[k] = ("EMPTY",neighbours[k])
            else:
                neighbours[k] = (self.env[neighbours[k]].get_type(),neighbours[k])
        return neighbours

    def get_grid(self) -> List[List[int]]:
        """
        Return a nice representation of the board.

        Returns:
            str: The nice representation of the board.
        """
        grid_data = [[0 for _ in range(self.dimensions[1])] for _ in range(self.dimensions[0])]
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if (i,j) in self.env:
                    piece_type = self.env[(i,j)].get_type()
                    grid_data[i][j] = ("⬢", piece_type)
                else:
                    grid_data[i][j] = " "
         
        return grid_data
    
    def get_empty(self) -> Generator[Tuple[int, int], None, None]:
        """
        Returns a list of empty cells in the grid.

        Returns:
            List[Tuple[int, int]]: A list of tuples representing the coordinates of empty cells.
        """
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if (i,j) not in self.env:
                    yield (i,j)
       
    def to_json(self) -> dict:
        """
        Converts the board to a JSON object.

        Returns:
            dict: The JSON representation of the board.
        """
        return {"env":{str(x):y for x,y in self.env.items()},"dim":self.dimensions}

    @classmethod
    def from_json(cls, data) -> Serializable:
        d = json.loads(data)
        dd = json.loads(data)
        for x,y in d["env"].items():
            # TODO eval is unsafe
            del dd["env"][x]
            dd["env"][eval(x)] = Piece.from_json(json.dumps(y))
        return cls(**dd)
