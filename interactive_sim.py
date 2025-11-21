#!/usr/bin/env python3
"""
Interactive car control simulator.

Controls:
- W/S: Increase/decrease steering (manual mode)
- A/D: Steering left/right (manual mode)
- SPACE: Toggle manual/auto control
- R: Reset/restart
- 1-9: Select controller
- ESC: Quit
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pygame
from collections import deque

from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, State,
    CONTROL_START_IDX, STEER_RANGE
)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)

# Display settings
WIDTH, HEIGHT = 1400, 900
GRAPH_WIDTH = 1000
GRAPH_HEIGHT = 200
INFO_WIDTH = 380


class InteractiveSimulator:
    """Interactive car control simulator with manual and automatic modes."""
    
    def __init__(self, model_path: str, data_path: str, controllers: list):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("TinyPhysics Interactive Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        
        # Load model and data
        self.model_path = model_path
        self.data_path = data_path
        self.available_controllers = controllers
        self.current_controller_idx = 0
        
        # Simulation state
        self.manual_mode = True
        self.manual_steer = 0.0
        self.paused = False
        self.speed = 1.0  # Simulation speed multiplier
        
        # History for graphs
        self.history_length = 200
        self.target_history = deque(maxlen=self.history_length)
        self.current_history = deque(maxlen=self.history_length)
        self.steer_history = deque(maxlen=self.history_length)
        self.step_history = deque(maxlen=self.history_length)
        
        # Initialize simulation
        self.reset_simulation()
        
    def reset_simulation(self):
        """Reset simulation to initial state."""
        # Load controller
        controller_name = self.available_controllers[self.current_controller_idx]
        controller_class = importlib.import_module(f'controllers.{controller_name}').Controller
        self.controller = controller_class()
        self.controller_name = controller_name
        
        # Load model
        model = TinyPhysicsModel(self.model_path, debug=False)
        
        # Create simulator
        self.sim = TinyPhysicsSimulator(
            model,
            self.data_path,
            controller=self.controller,
            debug=False,
            trace_logger=None,
        )
        
        # Reset histories
        self.target_history.clear()
        self.current_history.clear()
        self.steer_history.clear()
        self.step_history.clear()
        
        self.manual_steer = 0.0
        
    def handle_input(self):
        """Handle keyboard input."""
        keys = pygame.key.get_pressed()
        
        # Manual steering control
        if self.manual_mode:
            steer_rate = 0.05
            if keys[pygame.K_a]:
                self.manual_steer = max(STEER_RANGE[0], self.manual_steer - steer_rate)
            if keys[pygame.K_d]:
                self.manual_steer = min(STEER_RANGE[1], self.manual_steer + steer_rate)
            if keys[pygame.K_w]:
                self.manual_steer *= 0.95  # Decay toward zero
            if keys[pygame.K_s]:
                self.manual_steer = 0.0  # Reset to zero
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # Toggle manual/auto
                if event.key == pygame.K_SPACE:
                    self.manual_mode = not self.manual_mode
                    if not self.manual_mode:
                        self.manual_steer = 0.0
                
                # Reset
                if event.key == pygame.K_r:
                    self.reset_simulation()
                
                # Pause
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                
                # Speed control
                if event.key == pygame.K_LEFTBRACKET:
                    self.speed = max(0.25, self.speed * 0.5)
                if event.key == pygame.K_RIGHTBRACKET:
                    self.speed = min(4.0, self.speed * 2.0)
                
                # Controller selection
                if event.key >= pygame.K_1 and event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(self.available_controllers):
                        self.current_controller_idx = idx
                        self.reset_simulation()
                
                # Quit
                if event.key == pygame.K_ESCAPE:
                    return False
        
        return True
    
    def update(self):
        """Update simulation state."""
        if self.paused or self.sim.step_idx >= len(self.sim.data):
            return
        
        # Override action if in manual mode
        if self.manual_mode and self.sim.step_idx >= CONTROL_START_IDX:
            # Manually control
            self.sim.action_history[-1] = self.manual_steer
        
        # Step simulation
        self.sim.step()
        
        # Record history
        self.target_history.append(self.sim.target_lataccel_history[-1])
        self.current_history.append(self.sim.current_lataccel_history[-1])
        self.steer_history.append(self.sim.action_history[-1])
        self.step_history.append(self.sim.step_idx)
    
    def draw_graph(self, x, y, width, height, data1, data2, label1, label2, color1, color2, y_range=(-5, 5)):
        """Draw a line graph."""
        # Background
        pygame.draw.rect(self.screen, GRAY, (x, y, width, height), 1)
        
        # Draw zero line
        zero_y = y + height // 2
        pygame.draw.line(self.screen, GRAY, (x, zero_y), (x + width, zero_y), 1)
        
        # Draw control start line
        if len(self.step_history) > 0:
            steps = list(self.step_history)
            if steps[0] <= CONTROL_START_IDX <= steps[-1]:
                idx = steps.index(CONTROL_START_IDX) if CONTROL_START_IDX in steps else 0
                control_x = x + int((idx / len(steps)) * width)
                pygame.draw.line(self.screen, YELLOW, (control_x, y), (control_x, y + height), 2)
        
        # Draw data
        if len(data1) > 1:
            points1 = []
            for i, val in enumerate(data1):
                px = x + (i / (self.history_length - 1)) * width
                normalized = (val - y_range[0]) / (y_range[1] - y_range[0])
                py = y + height - (normalized * height)
                py = max(y, min(y + height, py))
                points1.append((px, py))
            if len(points1) > 1:
                pygame.draw.lines(self.screen, color1, False, points1, 2)
        
        if len(data2) > 1:
            points2 = []
            for i, val in enumerate(data2):
                px = x + (i / (self.history_length - 1)) * width
                normalized = (val - y_range[0]) / (y_range[1] - y_range[0])
                py = y + height - (normalized * height)
                py = max(y, min(y + height, py))
                points2.append((px, py))
            if len(points2) > 1:
                pygame.draw.lines(self.screen, color2, False, points2, 2)
        
        # Labels
        label1_text = self.small_font.render(label1, True, color1)
        label2_text = self.small_font.render(label2, True, color2)
        self.screen.blit(label1_text, (x + 10, y + 5))
        self.screen.blit(label2_text, (x + 10, y + 25))
        
        # Y-axis labels
        y_min_text = self.small_font.render(f"{y_range[0]:.1f}", True, WHITE)
        y_max_text = self.small_font.render(f"{y_range[1]:.1f}", True, WHITE)
        self.screen.blit(y_min_text, (x - 40, y + height - 10))
        self.screen.blit(y_max_text, (x - 40, y))
    
    def draw_info_panel(self):
        """Draw information panel."""
        x = WIDTH - INFO_WIDTH + 20
        y = 20
        line_height = 30
        
        def draw_text(text, color=WHITE, offset=0):
            nonlocal y
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (x + offset, y))
            y += line_height
        
        # Title
        draw_text("TinyPhysics Simulator", CYAN)
        y += 10
        
        # Mode
        mode_text = "MANUAL" if self.manual_mode else "AUTO"
        mode_color = ORANGE if self.manual_mode else GREEN
        draw_text(f"Mode: {mode_text}", mode_color)
        
        # Controller
        draw_text(f"Controller: {self.controller_name}", YELLOW)
        
        y += 10
        
        # Current state
        if self.sim.step_idx < len(self.sim.data):
            state = self.sim.state_history[-1]
            draw_text(f"Step: {self.sim.step_idx}/{len(self.sim.data)}")
            draw_text(f"v_ego: {state.v_ego:.1f} m/s")
            draw_text(f"a_ego: {state.a_ego:.2f} m/s²")
            draw_text(f"roll: {state.roll_lataccel:.2f} m/s²")
            
            y += 10
            
            # Control values
            target = self.sim.target_lataccel_history[-1]
            current = self.sim.current_lataccel_history[-1]
            steer = self.sim.action_history[-1]
            error = target - current
            
            draw_text(f"Target: {target:.3f} m/s²", GREEN)
            draw_text(f"Current: {current:.3f} m/s²", BLUE)
            draw_text(f"Error: {error:.3f} m/s²", RED if abs(error) > 0.5 else WHITE)
            draw_text(f"Steer: {steer:.3f}", ORANGE)
            
            y += 10
            
            # Cost
            if self.sim.step_idx >= CONTROL_START_IDX:
                cost = self.sim.compute_cost()
                draw_text(f"Lataccel Cost: {cost['lataccel_cost']:.2f}")
                draw_text(f"Jerk Cost: {cost['jerk_cost']:.2f}")
                draw_text(f"Total Cost: {cost['total_cost']:.2f}")
        
        # Controls help
        y = HEIGHT - 250
        draw_text("CONTROLS:", CYAN)
        self.screen.blit(self.small_font.render("SPACE: Toggle Auto/Manual", True, WHITE), (x, y + 30))
        self.screen.blit(self.small_font.render("A/D: Steer left/right", True, WHITE), (x, y + 50))
        self.screen.blit(self.small_font.render("W: Decay steer", True, WHITE), (x, y + 70))
        self.screen.blit(self.small_font.render("S: Reset steer", True, WHITE), (x, y + 90))
        self.screen.blit(self.small_font.render("R: Reset simulation", True, WHITE), (x, y + 110))
        self.screen.blit(self.small_font.render("P: Pause", True, WHITE), (x, y + 130))
        self.screen.blit(self.small_font.render("1-9: Select controller", True, WHITE), (x, y + 150))
        self.screen.blit(self.small_font.render("ESC: Quit", True, WHITE), (x, y + 170))
    
    def draw(self):
        """Render everything."""
        self.screen.fill(BLACK)
        
        # Draw graphs
        margin = 20
        graph_y_start = 50
        
        # Lateral acceleration
        self.draw_graph(
            margin, graph_y_start,
            GRAPH_WIDTH, GRAPH_HEIGHT,
            self.target_history, self.current_history,
            "Target Lataccel", "Current Lataccel",
            GREEN, BLUE,
            y_range=(-3, 3)
        )
        
        # Steering command
        self.draw_graph(
            margin, graph_y_start + GRAPH_HEIGHT + 40,
            GRAPH_WIDTH, GRAPH_HEIGHT,
            self.steer_history, [],
            "Steering Command", "",
            ORANGE, ORANGE,
            y_range=(-2.5, 2.5)
        )
        
        # Error
        if len(self.target_history) > 0 and len(self.current_history) > 0:
            error_history = deque([t - c for t, c in zip(self.target_history, self.current_history)], 
                                 maxlen=self.history_length)
            self.draw_graph(
                margin, graph_y_start + 2 * (GRAPH_HEIGHT + 40),
                GRAPH_WIDTH, GRAPH_HEIGHT,
                error_history, [],
                "Tracking Error", "",
                RED, RED,
                y_range=(-2, 2)
            )
        
        # Info panel
        self.draw_info_panel()
        
        # Status text at bottom
        status_text = "PAUSED" if self.paused else f"Speed: {self.speed}x"
        status_color = YELLOW if self.paused else WHITE
        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (margin, HEIGHT - 30))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_input()
            
            # Update at target speed
            for _ in range(int(self.speed)):
                if not self.paused:
                    self.update()
            
            self.draw()
            self.clock.tick(10)  # 10 FPS
        
        pygame.quit()


def get_available_controllers():
    """Get list of available controllers."""
    return [f.stem for f in Path('controllers').iterdir() 
            if f.is_file() and f.suffix == '.py' and f.stem != '__init__']


def main():
    parser = argparse.ArgumentParser(description="Interactive car control simulator")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
    parser.add_argument("--data_path", type=str, default="./data/00000.csv")
    args = parser.parse_args()
    
    controllers = get_available_controllers()
    print(f"Available controllers: {controllers}")
    
    sim = InteractiveSimulator(args.model_path, args.data_path, controllers)
    sim.run()


if __name__ == "__main__":
    main()

