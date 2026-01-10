# ==================================================
# JOGO DA VELHA COM IA — DESAFIO COMPLETO
# IA: MINIMAX + NÍVEIS DE DIFICULDADE
# ==================================================

import random

# ----------------------------
# UTILIDADES DO JOGO
# ----------------------------

def print_board(board):
    print()
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")
    print()


def check_winner(board, player):
    win_positions = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    return any(all(board[i] == player for i in combo) for combo in win_positions)


def is_draw(board):
    return " " not in board


# ----------------------------
# IA — AVALIAÇÃO E MINIMAX
# ----------------------------

def evaluate(board):
    if check_winner(board, "O"):
        return 1
    if check_winner(board, "X"):
        return -1
    return 0


def minimax(board, is_maximizing, depth_limit=None, depth=0):
    score = evaluate(board)

    if score != 0 or is_draw(board):
        return score

    if depth_limit is not None and depth >= depth_limit:
        return 0  # limitação para nível médio

    if is_maximizing:
        best = -float("inf")
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                best = max(best, minimax(board, False, depth_limit, depth + 1))
                board[i] = " "
        return best
    else:
        best = float("inf")
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                best = min(best, minimax(board, True, depth_limit, depth + 1))
                board[i] = " "
        return best


def best_move(board, difficulty):
    empty_positions = [i for i in range(9) if board[i] == " "]

    # ----------------------------
    # FÁCIL — ALEATÓRIO
    # ----------------------------
    if difficulty == "1":
        return random.choice(empty_positions)

    # ----------------------------
    # MÉDIO — MINIMAX LIMITADO
    # ----------------------------
    depth_limit = 2 if difficulty == "2" else None

    best_score = -float("inf")
    move = None

    for i in empty_positions:
        board[i] = "O"
        score = minimax(board, False, depth_limit)
        board[i] = " "
        if score > best_score:
            best_score = score
            move = i

    return move


# ----------------------------
# JOGO HUMANO VS IA
# ----------------------------

def choose_difficulty():
    print("Escolha a dificuldade:")
    print("1 - Fácil")
    print("2 - Médio")
    print("3 - Difícil (IA impossível)")

    while True:
        choice = input("Digite 1, 2 ou 3: ")
        if choice in ["1", "2", "3"]:
            return choice
        print("⚠️ Opção inválida.")


def play_vs_ai():
    board = [" " for _ in range(9)]
    difficulty = choose_difficulty()

    while True:
        print_board(board)

        # -------- TRATAMENTO DE EXCEÇÕES --------
        try:
            move = int(input("Sua jogada (0-8): "))

            if move < 0 or move > 8:
                print("⚠️ Digite um número entre 0 e 8.")
                continue

            if board[move] != " ":
                print("⚠️ Essa posição já está ocupada.")
                continue

        except ValueError:
            print("⚠️ Digite apenas números.")
            continue
        # ---------------------------------------

        board[move] = "X"

        if check_winner(board, "X"):
            print_board(board)
            print("🎉 Você venceu!")
            break

        if is_draw(board):
            print_board(board)
            print("😐 Empate!")
            break

        print("🤖 IA está pensando...")
        ai_move = best_move(board, difficulty)
        board[ai_move] = "O"

        if check_winner(board, "O"):
            print_board(board)
            print("🤖 IA venceu!")
            break

        if is_draw(board):
            print_board(board)
            print("😐 Empate!")
            break


# ----------------------------
# EXECUÇÃO
# ----------------------------

play_vs_ai()
