"""Arquivo principal do jogo de damas."""

import pygame
import sys
from typing import Optional
from config import WindowConfig, BoardConfig, ColorsConfig, UIElementConfig
from core.enums import PlayerColor, GameMode, Difficulty
from core.game_manager import GameManager
from core.position import Position
from renderers.board_renderer import BoardRenderer
from renderers.piece_renderer import PieceRenderer
from ui.mode_selector import ModeSelector
from ui.difficulty_selector import DifficultySelector
from ui.evaluator_selector import EvaluatorSelector
from ui.control_panel import ControlPanel
from ui.status_panel import StatusPanel
from ui.scrollable_panel import ScrollablePanel
import evaluators


class CheckersWindow:
    """Janela principal do jogo de damas."""

    def __init__(self):
        """Inicializa a aplicação."""
        pygame.init()
        pygame.font.init()

        # Configurar janela
        self.width = WindowConfig.WIDTH
        self.height = WindowConfig.HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(WindowConfig.TITLE)

        # Clock para FPS
        self.clock = pygame.time.Clock()
        self.running = True

        # Criar gerenciador do jogo
        self.game_manager = GameManager(GameMode.HUMAN_VS_AI, Difficulty.MEDIUM)

        # Criar renderizadores
        self.board_renderer = BoardRenderer(self.screen)
        self.piece_renderer = PieceRenderer(self.screen)

        # Criar componentes UI
        # Layout: Coluna esquerda com seletores e controles
        left_column_x = 20
        # Largura dos botões: do x=20 até 20px antes do tabuleiro (que está em x=260)
        button_width = BoardConfig.BOARD_X - 20 - left_column_x  # 260 - 20 - 20 = 220

        # Alinhar com a borda superior do tabuleiro
        selector_y_start = BoardConfig.BOARD_Y

        # Criar componentes com suas posições originais
        self.mode_selector = ModeSelector(
            x=left_column_x,
            y=selector_y_start,
            width=button_width,
            on_mode_change=self._on_mode_change
        )

        difficulty_y = selector_y_start + 3 * UIElementConfig.SELECTOR_BUTTON_HEIGHT + 30
        self.difficulty_selector = DifficultySelector(
            x=left_column_x,
            y=difficulty_y,
            width=button_width,
            on_difficulty_change=self._on_difficulty_change
        )

        num_evaluators = len(evaluators.get_evaluator_names())
        evaluator_y = difficulty_y + 3 * UIElementConfig.SELECTOR_BUTTON_HEIGHT + 30
        self.red_evaluator_selector = EvaluatorSelector(
            x=left_column_x,
            y=evaluator_y,
            width=button_width,
            title="AVALIADOR RED",
            on_evaluator_change=self._on_red_evaluator_change,
            default_evaluator=evaluators.get_default_evaluator_name()
        )

        black_evaluator_y = evaluator_y + num_evaluators * UIElementConfig.SELECTOR_BUTTON_HEIGHT + 30
        self.black_evaluator_selector = EvaluatorSelector(
            x=left_column_x,
            y=black_evaluator_y,
            width=button_width,
            title="AVALIADOR BLACK",
            on_evaluator_change=self._on_black_evaluator_change,
            default_evaluator="Meninas Superpoderosas"
        )

        control_y = black_evaluator_y + num_evaluators * UIElementConfig.SELECTOR_BUTTON_HEIGHT + 30
        self.control_panel = ControlPanel(
            x=left_column_x,
            y=control_y,
            width=button_width,
            on_reset=self._on_reset_clicked
        )

        status_y = control_y + self.control_panel.panel_height + 30
        self.status_panel = StatusPanel(
            x=left_column_x,
            y=status_y,
            width=button_width,
            game_manager=self.game_manager
        )

        # Criar painel rolável para os componentes do menu lateral
        # Altura disponível: da posição inicial até o final da janela
        panel_height = WindowConfig.HEIGHT - selector_y_start - 10
        self.scrollable_panel = ScrollablePanel(
            x=left_column_x,
            y=selector_y_start,
            width=button_width,
            height=panel_height,
            scroll_speed=30
        )

        # Adicionar componentes ao painel rolável
        self.scrollable_panel.add_component(
            self.mode_selector,
            lambda surface: self.mode_selector.render(surface),
            lambda mouse_pos: self.mode_selector.update(mouse_pos),
            lambda event: self.mode_selector.handle_event(event)
        )
        self.scrollable_panel.add_component(
            self.difficulty_selector,
            lambda surface: self.difficulty_selector.render(surface),
            lambda mouse_pos: self.difficulty_selector.update(mouse_pos),
            lambda event: self.difficulty_selector.handle_event(event)
        )
        self.scrollable_panel.add_component(
            self.red_evaluator_selector,
            lambda surface: self.red_evaluator_selector.render(surface),
            lambda mouse_pos: self.red_evaluator_selector.update(mouse_pos),
            lambda event: self.red_evaluator_selector.handle_event(event)
        )
        self.scrollable_panel.add_component(
            self.black_evaluator_selector,
            lambda surface: self.black_evaluator_selector.render(surface),
            lambda mouse_pos: self.black_evaluator_selector.update(mouse_pos),
            lambda event: self.black_evaluator_selector.handle_event(event)
        )
        self.scrollable_panel.add_component(
            self.control_panel,
            lambda surface: self.control_panel.render(surface),
            lambda mouse_pos: self.control_panel.update(mouse_pos),
            lambda event: self.control_panel.handle_event(event)
        )
        self.scrollable_panel.add_component(
            self.status_panel,
            lambda surface: self.status_panel.render(surface),
            lambda mouse_pos: self.status_panel.update(mouse_pos),
            lambda event: self.status_panel.handle_event(event)
        )

    def _on_mode_change(self, mode: GameMode) -> None:
        """
        Callback para mudança de modo de jogo.

        Args:
            mode: Novo modo de jogo
        """
        print(f"Modo alterado para: {mode.value}")
        self.game_manager.set_game_mode(mode)
        self.difficulty_selector.update_visibility(mode)
        self.red_evaluator_selector.update_visibility(mode)
        self.black_evaluator_selector.update_visibility(mode)

    def _on_difficulty_change(self, difficulty: Difficulty) -> None:
        """
        Callback para mudança de dificuldade.

        Args:
            difficulty: Nova dificuldade
        """
        print(f"Dificuldade alterada para: {difficulty.value}")
        self.game_manager.set_difficulty(difficulty)

    def _on_red_evaluator_change(self, evaluator_name: str) -> None:
        """
        Callback para mudança de avaliador RED.

        Args:
            evaluator_name: Nome do novo avaliador
        """
        print(f"Avaliador RED alterado para: {evaluator_name}")
        self.game_manager.set_red_evaluator(evaluator_name)

    def _on_black_evaluator_change(self, evaluator_name: str) -> None:
        """
        Callback para mudança de avaliador BLACK.

        Args:
            evaluator_name: Nome do novo avaliador
        """
        print(f"Avaliador BLACK alterado para: {evaluator_name}")
        self.game_manager.set_black_evaluator(evaluator_name)

    def _on_reset_clicked(self) -> None:
        """Callback para quando clicar em "Reiniciar"."""
        self.game_manager.reset_game()
        print("Jogo reiniciado")

    def _handle_board_click(self, event: pygame.event.Event) -> None:
        """
        Processa clique no tabuleiro.

        Args:
            event: Evento do pygame
        """
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return

        # Verificar se é turno humano
        if not self.game_manager.is_human_turn():
            return

        # Obter posição clicada
        mouse_x, mouse_y = event.pos
        square = BoardConfig.get_square_from_pos(mouse_x, mouse_y)

        if square is None:
            return

        row, col = square
        clicked_pos = Position(row, col)

        # Lógica de seleção e movimento
        if self.game_manager.selected_piece is None:
            # Nenhuma peça selecionada: tentar selecionar
            if self.game_manager.select_piece(clicked_pos):
                print(f"Peça selecionada em {clicked_pos}")
        else:
            # Peça já selecionada
            if clicked_pos == self.game_manager.selected_piece:
                # Clicou na mesma peça: desselecionar
                self.game_manager.deselect_piece()
                print("Peça desselecionada")
            elif self.game_manager.make_human_move(clicked_pos):
                # Movimento executado com sucesso
                print(f"Movimento para {clicked_pos}")
            else:
                # Tentar selecionar outra peça
                self.game_manager.deselect_piece()
                if self.game_manager.select_piece(clicked_pos):
                    print(f"Peça selecionada em {clicked_pos}")

    def handle_events(self) -> None:
        """Processa eventos do pygame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Processar eventos do painel rolável
            self.scrollable_panel.handle_event(event)

            # Processar cliques no tabuleiro
            self._handle_board_click(event)

    def update(self) -> None:
        """Atualiza lógica da aplicação."""
        current_time = pygame.time.get_ticks()
        mouse_pos = pygame.mouse.get_pos()

        # Atualizar gerenciador de estado (processa IA se necessário)
        self.game_manager.update(current_time)

        # Atualizar painel rolável
        self.scrollable_panel.update(mouse_pos)

    def render(self) -> None:
        """Renderiza todos os elementos na tela."""
        # Limpar tela
        self.screen.fill(ColorsConfig.BACKGROUND)

        # Renderizar tabuleiro
        self.board_renderer.render(
            self.game_manager.board,
            self.game_manager.selected_piece,
            self.game_manager.valid_moves_for_selected
        )

        # Renderizar peças
        self.piece_renderer.render_all_pieces(self.game_manager.board)

        # Renderizar painel rolável com componentes UI
        self.scrollable_panel.render(self.screen)

        # Renderizar mensagem se IA estiver pensando
        if self.game_manager.is_ai_thinking:
            self._render_thinking_message()

        # Atualizar tela
        pygame.display.flip()

    def _render_thinking_message(self) -> None:
        """Renderiza mensagem de IA pensando."""
        font = pygame.font.Font(None, 24)
        text_surface = font.render("IA está pensando...", True, ColorsConfig.TEXT_SECONDARY)
        text_rect = text_surface.get_rect(
            center=(BoardConfig.BOARD_X + BoardConfig.BOARD_SIZE // 2, BoardConfig.BOARD_Y + BoardConfig.BOARD_SIZE + 25)
        )
        self.screen.blit(text_surface, text_rect)

    def run(self) -> None:
        """Loop principal do jogo."""
        while self.running:
            # Processar eventos
            self.handle_events()

            # Atualizar
            self.update()

            # Renderizar
            self.render()

            # Manter FPS
            self.clock.tick(WindowConfig.FPS)

        # Encerrar pygame
        pygame.quit()
        sys.exit()


def main():
    """Função principal."""
    game = CheckersWindow()
    game.run()


if __name__ == "__main__":
    main()
