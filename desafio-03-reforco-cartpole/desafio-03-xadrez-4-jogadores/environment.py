import random

class Xadrez4JogadoresEnv:
    def __init__(self, size=6):
        self.size = size
        self.players = ["♔", "♕", "♖", "♜"]

        self.reset()

    def reset(self):
        # Tabuleiro vazio
        self.board = [["." for _ in range(self.size)] for _ in range(self.size)]

        # Posições iniciais fixas (uma peça por jogador)
        self.positions = {
            "♔": (0, 0),
            "♕": (0, self.size - 1),
            "♖": (self.size - 1, 0),
            "♜": (self.size - 1, self.size - 1),
        }


        for p, (x, y) in self.positions.items():
            self.board[x][y] = p

        self.done = False
        return self.get_state()

    def get_state(self):
        # Estado simples: posições de todos os jogadores
        return tuple(self.positions[p] for p in self.players)

    def print_board(self):
        print()
        for row in self.board:
            print(" ".join(row))
        print()

    def step(self, player, action):
        """
        action: 0 = cima, 1 = baixo, 2 = esquerda, 3 = direita
        """
        if self.done:
            return self.get_state(), 0, True

        dx, dy = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }[action]

        x, y = self.positions[player]
        nx, ny = x + dx, y + dy

        # Movimento fora do tabuleiro
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return self.get_state(), -0.1, False

        reward = 0

        # Verifica se capturou alguém
        for other, pos in self.positions.items():
            if other != player and pos == (nx, ny):
                reward = 1
                self.done = True
                print(f"🏆 Jogador {player} capturou {other}!")
                break

        # Atualiza tabuleiro
        self.board[x][y] = "."
        self.board[nx][ny] = player
        self.positions[player] = (nx, ny)

        return self.get_state(), reward, self.done
