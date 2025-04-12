import random
import json
from collections import defaultdict
import time
import pygame
import sys
from datetime import datetime
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Initialisation de Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maître du Temps - Édition ML Ultime")

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 120, 215)
GOLD = (255, 215, 0)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (100, 100, 100)
DARK_BLUE = (30, 30, 60)
LIGHT_BLUE = (70, 130, 180)
NEON_BLUE = (30, 144, 255)
CYAN = (0, 255, 255)
PURPLE = (138, 43, 226)
GRADIENT_BG = [(0, 0, 50), (0, 0, 70), (0, 0, 90)]

# Polices
font_large = pygame.font.SysFont('Arial', 40)
font_medium = pygame.font.SysFont('Arial', 30)
font_small = pygame.font.SysFont('Arial', 20)

# Icônes d'actions
ACTION_ICONS = {
    "Attaquer": "⚔",
    "Défendre": "🛡",
    "Collecter": "📦"
}


class BasicAI:
    """IA de base pour le niveau facile"""

    def __init__(self, zones, actions):
        self.zones = zones
        self.actions = actions

    def choose_action(self, history):
        return random.choice(self.zones), random.choice(self.actions)


class QLearningAI:
    """IA avec Q-Learning pour le niveau moyen"""

    def __init__(self, zones, actions):
        self.zones = zones
        self.actions = actions
        self.q_table = defaultdict(lambda: np.zeros(len(zones) * len(actions)))
        self.alpha = 0.1  # Taux d'apprentissage
        self.gamma = 0.6  # Facteur de discount
        self.epsilon = 0.2  # Exploration vs exploitation
        self.last_state = None
        self.last_action = None

    def get_state_key(self, history):
        """Convertit l'historique en une clé d'état"""
        if not history:
            return "init"
        last_turn = history[-1]
        return f"{last_turn['player'][0]}_{last_turn['player'][1]}"

    def choose_action(self, history):
        state = self.get_state_key(history)
        self.last_state = state

        # Exploration aléatoire
        if np.random.uniform(0, 1) < self.epsilon:
            zone = random.choice(self.zones)
            action = random.choice(self.actions)
            self.last_action = (zone, action)
            return zone, action

        # Exploitation: choisir la meilleure action selon Q-table
        q_values = self.q_table[state]
        best_idx = np.argmax(q_values)
        zone_idx = best_idx // len(self.actions)
        action_idx = best_idx % len(self.actions)
        self.last_action = (self.zones[zone_idx], self.actions[action_idx])
        return self.last_action

    def learn(self, history, reward):
        if self.last_state is None or self.last_action is None:
            return

        state = self.last_state
        zone, act = self.last_action
        zone_idx = self.zones.index(zone)
        act_idx = self.actions.index(act)
        action_idx = zone_idx * len(self.actions) + act_idx

        next_state = self.get_state_key(history)

        # Mise à jour Q-value
        old_value = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action_idx] = new_value

    def save(self, filename):
        """Sauvegarde la Q-table"""
        with open(filename, 'wb') as f:
            joblib.dump(dict(self.q_table), f)

    def load(self, filename):
        """Charge la Q-table"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                q_table = joblib.load(f)
                self.q_table.update(q_table)


class DQN(nn.Module):
    """Réseau de neurones pour le Deep Q-Learning"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQLearningAI:
    """IA avec Deep Q-Learning pour le niveau difficile"""

    def __init__(self, zones, actions):
        self.zones = zones
        self.actions = actions
        self.input_size = 20  # Taille de l'état encodé
        self.output_size = len(zones) * len(actions)
        self.model = DQN(self.input_size, self.output_size)
        self.target_model = DQN(self.input_size, self.output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 0.3
        self.update_target_every = 10
        self.steps = 0

    def encode_state(self, history):
        """Encode l'historique en vecteur numérique"""
        state = np.zeros(self.input_size)
        if history:
            last = history[-1]
            # Encodage de la dernière action du joueur
            if 'player' in last:
                zone_idx = self.zones.index(last['player'][0])
                act_idx = self.actions.index(last['player'][1])
                state[zone_idx] = 1
                state[len(self.zones) + act_idx] = 1

            # Encodage du score relatif
            if len(history) > 1:
                state[-1] = history[-1]['result']['ai_points'] - history[-1]['result']['player_points']

        return torch.FloatTensor(state)

    def choose_action(self, history):
        if np.random.random() < self.epsilon:
            return (random.choice(self.zones), random.choice(self.actions))

        state = self.encode_state(history).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        action_idx = torch.argmax(q_values).item()
        zone_idx = action_idx // len(self.actions)
        act_idx = action_idx % len(self.actions)
        return (self.zones[zone_idx], self.actions[act_idx])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Convertir les actions en indices
        action_indices = []
        for a in actions:
            zone, act = a
            zone_idx = self.zones.index(zone)
            act_idx = self.actions.index(act)
            action_idx = zone_idx * len(self.actions) + act_idx
            action_indices.append(action_idx)
        action_indices = torch.LongTensor(action_indices).unsqueeze(1)

        # Calcul des Q-values actuelles
        current_q = self.model(states).gather(1, action_indices)

        # Calcul des Q-values cibles
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calcul de la perte
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Rétropropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour périodique du modèle cible
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        """Charge le modèle"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class HybridAI:
    """IA hybride pour le niveau expert (combinaison de plusieurs techniques)"""
    def __init__(self, zones, actions):
        self.zones = zones
        self.actions = actions
        self.q_learning = QLearningAI(zones, actions)
        self.deep_q = DeepQLearningAI(zones, actions)
        self.mode = "deep"  # Commence en mode Deep Q-Learning

    def encode_state(self, history):
        """Délègue l'encodage d'état au Deep Q-Learning"""
        return self.deep_q.encode_state(history)

    def remember(self, state, action, reward, next_state, done):
        """Délègue la mémorisation au Deep Q-Learning"""
        self.deep_q.remember(state, action, reward, next_state, done)

    def choose_action(self, history):
        # Alterne entre les deux modes selon les performances
        if random.random() < 0.7:
            return self.deep_q.choose_action(history)
        else:
            return self.q_learning.choose_action(history)

    def learn(self, history, action, reward):
        state = self.encode_state(history[:-1]) if len(history) > 1 else None
        next_state = self.encode_state(history)

        # Apprentissage pour les deux modèles
        self.remember(state, action, reward, next_state, False)
        self.deep_q.train()

        self.q_learning.learn(history, reward)

    def save(self, filename_prefix):
        self.deep_q.save(f"{filename_prefix}_deep.pth")
        self.q_learning.save(f"{filename_prefix}_q.pkl")

    def load(self, filename_prefix):
        self.deep_q.load(f"{filename_prefix}_deep.pth")
        self.q_learning.load(f"{filename_prefix}_q.pkl")

class Game:
    def __init__(self):
        self.zones = ["A", "B", "C", "D", "E"]
        self.actions = ["Collecter", "Attaquer", "Défendre"]
        self.items = {
            "Bouclier": {"cost": 3, "effect": "block_attack", "icon": "🛡️"},
            "Scanner": {"cost": 2, "effect": "reveal_ai", "icon": "🔍"},
            "Piège": {"cost": 4, "effect": "trap_zone", "icon": "💣"},
            "Soin": {"cost": 3, "effect": "+2 pts", "icon": "❤️"}
        }
        self.reset_game()
        self.highscores = self.load_scores()

        # Chargement des modèles IA
        self.load_ai_models()

    def load_ai_models(self):
        """Initialise les modèles d'IA pour chaque niveau"""
        self.ai_models = {
            1: BasicAI(self.zones, self.actions),  # Niveau facile
            2: QLearningAI(self.zones, self.actions),  # Niveau moyen
            3: DeepQLearningAI(self.zones, self.actions),  # Niveau difficile
            4: HybridAI(self.zones, self.actions)  # Niveau expert
        }

        # Charge les modèles sauvegardés s'ils existent
        if not os.path.exists('ai_models'):
            os.makedirs('ai_models')

        for level in self.ai_models:
            if hasattr(self.ai_models[level], 'load'):
                self.ai_models[level].load(f'ai_models/level_{level}.pth')

    def save_ai_models(self):
        """Sauvegarde les modèles d'IA"""
        for level in self.ai_models:
            if hasattr(self.ai_models[level], 'save'):
                self.ai_models[level].save(f'ai_models/level_{level}.pth')

    def load_scores(self):
        try:
            with open('highscores.json', 'r') as f:
                scores = json.load(f)
                # Ensure all scores have the 'level' field
                for score in scores:
                    if 'level' not in score:
                        score['level'] = 1  # Default level for old scores
                return scores
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_score(self, score, level):
        self.highscores.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "score": score,
            "level": level
        })
        # Keep only the last 20 scores to prevent file from growing too large
        self.highscores = self.highscores[-20:]
        with open('highscores.json', 'w') as f:
            json.dump(self.highscores, f)

    def export_joblib(self, filename="game.joblib"):
        data = {
            'metadata': {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'version': '1.0',
                'game': 'Maître du Temps ML'
            },
            'scores': {
                'player': self.player_score,
                'ai': self.ai_score
            },
            'history': self.history,
            'config': {
                'ai_level': self.ai_level,
                'used_items': self.player_inventory
            }
        }
        joblib.dump(data, filename)

    def import_joblib(self, filename="game.joblib"):
        data = joblib.load(filename)
        self.player_score = data['scores']['player']
        self.ai_score = data['scores']['ai']
        self.history = data['history']
        self.ai_level = data['config']['ai_level']
        self.player_inventory = data['config']['used_items']
        return data

    def resolve_turn(self, player_zone, player_action, ai_zone, ai_action):
        zone_e_bonus = 0
        if player_zone == "E" and self.current_event == "Bonus":
            zone_e_bonus = 1

        if player_zone != ai_zone:
            player_pts = (1 if player_action == "Collecter" else 0) + zone_e_bonus
            ai_pts = 1 if ai_action == "Collecter" else 0
            return {"player_points": player_pts, "ai_points": ai_pts}
        else:
            if player_action == "Attaquer":
                if ai_action == "Défendre":
                    return {"player_points": 0 + zone_e_bonus, "ai_points": 1}
                else:
                    return {"player_points": 2 + zone_e_bonus, "ai_points": 0}
            elif player_action == "Défendre":
                if ai_action == "Attaquer":
                    return {"player_points": 1 + zone_e_bonus, "ai_points": 0}
                else:
                    return {"player_points": 0 + zone_e_bonus, "ai_points": 1 if ai_action == "Collecter" else 0}
            else:  # Collecter
                if ai_action == "Attaquer":
                    return {"player_points": 0 + zone_e_bonus, "ai_points": 2}
                elif ai_action == "Défendre":
                    return {"player_points": 1 + zone_e_bonus, "ai_points": 0}
                else:
                    return {"player_points": 1 + zone_e_bonus, "ai_points": 1}

    def reset_game(self):
        self.current_turn = 1
        self.player_score = 0
        self.ai_score = 0
        self.history = []
        self.ai_history = []
        self.player_inventory = []
        self.ai_level = 1
        self.current_event = None
        self.storm_zone = None
        self.time_loop_used = False
        self.mode = "menu"
        self.selected_zone = None
        self.selected_action = None
        self.last_ai_move = None
        self.show_ai_history = True

    def show_menu(self):
        screen.fill(BLUE)
        title = font_large.render("MAÎTRE DU TEMPS ML", True, GOLD)
        subtitle = font_medium.render("Édition Machine Learning", True, WHITE)

        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 160))

        pygame.draw.rect(screen, GREEN, (300, 250, 200, 50))
        play = font_medium.render("JOUER", True, BLACK)
        screen.blit(play, (400 - play.get_width() // 2, 265))

        pygame.draw.rect(screen, GOLD, (300, 320, 200, 50))
        scores = font_medium.render("SCORES", True, BLACK)
        screen.blit(scores, (400 - scores.get_width() // 2, 335))

        pygame.draw.rect(screen, RED, (300, 390, 200, 50))
        quit_text = font_medium.render("QUITTER", True, BLACK)
        screen.blit(quit_text, (400 - quit_text.get_width() // 2, 405))

        pygame.display.flip()

    def show_level_selection(self):
        screen.fill(BLUE)
        title = font_large.render("SELECTION DU NIVEAU", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        levels = [
            ("FACILE", 1, GREEN),
            ("MOYEN", 2, YELLOW),
            ("DIFFICILE", 3, ORANGE),
            ("EXPERT", 4, RED)
        ]

        for i, (text, level, color) in enumerate(levels):
            pygame.draw.rect(screen, color, (250, 200 + i * 80, 300, 60))
            level_text = font_medium.render(text, True, BLACK)
            screen.blit(level_text, (400 - level_text.get_width() // 2, 230 + i * 80))

        pygame.display.flip()

    def show_highscores(self):
        screen.fill(BLUE)
        title = font_large.render("MEILLEURS SCORES", True, GOLD)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))

        if not self.highscores:
            none_text = font_medium.render("Aucun score enregistré", True, WHITE)
            screen.blit(none_text, (WIDTH // 2 - none_text.get_width() // 2, 150))
        else:
            for i, score in enumerate(self.highscores[-10:]):
                # Handle old scores that might not have the 'level' field
                level = score.get('level', '?')
                text = f"{score['date']} - Niv.{level}: {score['score']} pts"
                score_text = font_small.render(text, True, WHITE)
                screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 150 + i * 30))

        back = font_medium.render("RETOUR (ESC)", True, WHITE)
        screen.blit(back, (WIDTH // 2 - back.get_width() // 2, HEIGHT - 50))
        pygame.display.flip()

    def show_ai_history_panel(self):
        if not self.show_ai_history or not self.ai_history:
            return

        panel_x, panel_y = 20, HEIGHT - 180
        width, height = 300, 160
        pygame.draw.rect(screen, DARK_BLUE, (panel_x, panel_y, width, height))
        pygame.draw.rect(screen, BLUE, (panel_x, panel_y, width, height), 2)

        title = font_small.render("Derniers coups IA:", True, WHITE)
        screen.blit(title, (panel_x + 10, panel_y + 10))

        for i, move in enumerate(reversed(self.ai_history[-5:])):
            y_pos = panel_y + 40 + i * 25
            color = RED if move.get("bluff", False) else WHITE

            text = f"T{move['turn']}: {move['zone']} {(move['action'])}"
            if i == 0:
                text = "▶ " + text
                color = YELLOW

            move_text = font_small.render(text, True, color)
            screen.blit(move_text, (panel_x + 20, y_pos))

    def show_game(self):
        screen.fill(BLUE)

        # Game info
        title = font_medium.render(f"Tour {self.current_turn}/10 - Niveau {self.ai_level}", True, WHITE)
        screen.blit(title, (20, 20))

        score = font_medium.render(f"Toi: {self.player_score} pts | IA: {self.ai_score} pts", True, WHITE)
        screen.blit(score, (WIDTH - score.get_width() - 20, 20))

        # Zones
        zone_width = 100
        for i, zone in enumerate(self.zones):
            x = 150 + i * (zone_width + 20)
            color = RED if zone == self.storm_zone else (GOLD if zone == self.selected_zone else BLUE)
            pygame.draw.rect(screen, color, (x, 200, zone_width, 100), 0 if zone == self.selected_zone else 2)
            zone_text = font_large.render(zone, True, WHITE if zone != self.selected_zone else BLACK)
            screen.blit(zone_text,
                        (x + zone_width // 2 - zone_text.get_width() // 2, 250 - zone_text.get_height() // 2))

        # Actions
        actions_y = 350
        for i, action in enumerate(self.actions):
            color = GREEN if action == self.selected_action else (GRAY if not self.selected_zone else GREEN)
            pygame.draw.rect(screen, color, (150 + i * 150, actions_y, 140, 50))
            action_text = font_medium.render(action, True, BLACK if action == self.selected_action else WHITE)
            screen.blit(action_text, (
                150 + i * 150 + 70 - action_text.get_width() // 2, actions_y + 25 - action_text.get_height() // 2))

        # Items
        items_y = 450
        for i, (item, details) in enumerate(self.items.items()):
            color = WHITE if self.player_score >= details["cost"] else GRAY
            pygame.draw.rect(screen, color, (150 + i * 120, items_y, 110, 40))

            item_text = font_small.render(f"{details['icon']} {item} ({details['cost']})", True, BLACK)
            screen.blit(item_text,
                        (150 + i * 120 + 55 - item_text.get_width() // 2, items_y + 20 - item_text.get_height() // 2))

        # Event
        if self.current_event:
            event_text = font_medium.render(f"Événement: {self.current_event}", True, WHITE)
            screen.blit(event_text, (WIDTH // 2 - event_text.get_width() // 2, 150))

        # Last AI move
        if self.last_ai_move:
            ai_zone, ai_action = self.last_ai_move
            move_text = font_medium.render(f"Dernier coup IA: {ai_zone} {ACTION_ICONS.get(ai_action, '')}", True,
                                           YELLOW)
            screen.blit(move_text, (WIDTH - move_text.get_width() - 20, 60))

        # AI history panel
        self.show_ai_history_panel()

        pygame.display.flip()

    def show_end_screen(self):
        self.save_ai_models()
        self.export_joblib()

        screen.fill(BLUE)
        title = font_large.render("FIN DE PARTIE", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        if self.player_score > self.ai_score:
            result = font_medium.render("Vous avez gagné !", True, GREEN)
        elif self.player_score < self.ai_score:
            result = font_medium.render("L'IA a gagné...", True, RED)
        else:
            result = font_medium.render("Égalité !", True, GOLD)

        screen.blit(result, (WIDTH // 2 - result.get_width() // 2, 200))

        score = font_medium.render(f"Score final: {self.player_score} - {self.ai_score}", True, WHITE)
        screen.blit(score, (WIDTH // 2 - score.get_width() // 2, 250))

        back = font_medium.render("RETOUR (ESC)", True, WHITE)
        screen.blit(back, (WIDTH // 2 - back.get_width() // 2, HEIGHT - 50))
        pygame.display.flip()

    def play_game(self):
        self.selected_zone = None
        self.selected_action = None

        while self.current_turn <= 10 and self.mode == "jeu":
            self.show_game()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_ai_models()
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.mode = "menu"
                        return
                    elif event.key == pygame.K_h:
                        self.show_ai_history = not self.show_ai_history

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()

                    # Zone selection
                    for i, zone in enumerate(self.zones):
                        zone_x = 150 + i * 120
                        if zone_x <= x <= zone_x + 100 and 200 <= y <= 300:
                            if not self.storm_zone or zone != self.storm_zone:
                                self.selected_zone = zone

                    # Action selection
                    for i, action in enumerate(self.actions):
                        action_x = 150 + i * 150
                        if action_x <= x <= action_x + 140 and 350 <= y <= 400:
                            if self.selected_zone:
                                self.selected_action = action

                    # Buy item
                    for i, item in enumerate(self.items):
                        item_x = 150 + i * 120
                        if item_x <= x <= item_x + 110 and 450 <= y <= 490:
                            if self.player_score >= self.items[item]["cost"]:
                                self.player_score -= self.items[item]["cost"]
                                self.player_inventory.append(item)

                    # Resolve turn
                    if self.selected_zone and self.selected_action:
                        # Choix de l'IA
                        ai_model = self.ai_models[self.ai_level]
                        ai_zone, ai_action = ai_model.choose_action(self.history)
                        self.last_ai_move = (ai_zone, ai_action)

                        # Enregistrement du coup de l'IA
                        self.ai_history.append({
                            "turn": self.current_turn,
                            "zone": ai_zone,
                            "action": ai_action,
                            "bluff": False
                        })

                        # Résolution du tour
                        result = self.resolve_turn(self.selected_zone, self.selected_action, ai_zone, ai_action)

                        self.player_score += result["player_points"]
                        self.ai_score += result["ai_points"]

                        # Enregistrement dans l'historique
                        turn_data = {
                            "turn": self.current_turn,
                            "player": (self.selected_zone, self.selected_action),
                            "ai": (ai_zone, ai_action),
                            "result": result
                        }
                        self.history.append(turn_data)

                        # Apprentissage pour l'IA
                        # In the play_game method, modify the learning section to:
                        if hasattr(ai_model, 'learn'):
                            reward = result["ai_points"] - result["player_points"]

                            if isinstance(ai_model, DeepQLearningAI):
                                state = ai_model.encode_state(self.history[:-1] if len(self.history) > 1 else [])
                                next_state = ai_model.encode_state(self.history)
                                ai_model.remember(state, (ai_zone, ai_action), reward, next_state, False)
                                ai_model.train()
                            elif isinstance(ai_model, HybridAI):
                                ai_model.learn(self.history, (ai_zone, ai_action), reward)
                            else:
                                ai_model.learn(self.history, reward)

                        self.current_turn += 1
                        self.selected_zone = None
                        self.selected_action = None

            pygame.display.flip()
            time.sleep(0.1)

        self.save_score(self.player_score, self.ai_level)
        self.mode = "fin"
        self.show_end_screen()

    def run(self):
        clock = pygame.time.Clock()

        while True:
            if self.mode == "menu":
                self.show_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.save_ai_models()
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        if 300 <= x <= 500 and 250 <= y <= 300:
                            self.mode = "selection"
                        elif 300 <= x <= 500 and 320 <= y <= 370:
                            self.mode = "scores"
                        elif 300 <= x <= 500 and 390 <= y <= 440:
                            self.save_ai_models()
                            pygame.quit()
                            sys.exit()

            elif self.mode == "selection":
                self.show_level_selection()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.save_ai_models()
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        for i in range(4):
                            if 250 <= x <= 550 and 200 + i * 80 <= y <= 260 + i * 80:
                                self.ai_level = i + 1
                                self.mode = "jeu"
                                self.play_game()

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.mode = "menu"

            elif self.mode == "scores":
                self.show_highscores()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.save_ai_models()
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.mode = "menu"

            elif self.mode == "fin":
                self.show_end_screen()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.save_ai_models()
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        self.reset_game()
                        self.mode = "menu"

            clock.tick(60)


if __name__ == "__main__":
    game = Game()
    game.run()
