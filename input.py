import pygame
import os

from utils import *


class InputManager:
    """
    Manages input from either a keyboard or a joystick for a given scene.

    This class initializes input handling for a game, allowing for input via
    keyboard or joystick. It automatically detects if a joystick is available
    at startup and loads any pre-configured joystick mappings from a config
    file.

    Attributes:
        joystick (pygame.joystick.Joystick or None): The joystick device, if one is connected.
        joystick_steer_axis (int or None): The joystick axis configured for steering.
        joystick_throttle_axis (int or None): The joystick axis configured for throttle.
        input_method (str): The current input method, either "keyboard" or "joystick".
        config_file (str): Path to the input configuration file.
    """

    def __init__(self):
        """
        Initializes the InputManager with the given scene, automatically detects
        joystick availability, and sets up the input method accordingly.
        """
        self.joystick = pygame.joystick.get_count() > 0
        self.joystick_config = {}
        self.input_method = "keyboard"
        self.config_folder = "settings"
        self.config_file = os.path.join(self.config_folder, "input_config.json")

        if self.joystick:
            print_formatted("Using Joystick Input", GREEN)
            self.input_method = "joystick"
            self.initialize_joystick()
        else:
            print_formatted("Using Keyboard Input", GREEN)

    def get_input(self):
        """
        Retrieves the current input based on the configured input method.

        Returns:
            tuple: A pair of (throttle, steer, brake) representing the current input state.
        """
        return self.get_joystick_control() if self.input_method == "joystick" else self.get_keyboard_control()

    def get_button(self, button_name):
        """
        Retrieves the current state of a joystick button based on the configured mapping.

        Parameters:
            button_name (str): The name of the button to retrieve.

        Returns:
            bool: True if the button is pressed, False otherwise.
        """
        if self.joystick:
            button = self.joystick_config.get(button_name)
            if button is not None:
                return self.joystick.get_button(button)
        return False

    def get_keyboard_control(self):
        """
        Retrieves keyboard inputs and translates them into throttle and steer commands.

        Returns:
            tuple: A pair of (throttle, steer) values derived from keyboard inputs.
        """
        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_UP] or keys[pygame.K_w]
        brake = keys[pygame.K_DOWN] or keys[pygame.K_s]
        steer = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])

        return steer, throttle, brake

    def initialize_joystick(self):
        if self.joystick:
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.joystick_config = self.load_joystick_mapping()

            if self.joystick_config == {}:
                print_formatted("Joystick detected, but no mapping found. Mapping joystick axes now.")
                self.joystick_config["steer"] = self.map_joystick_axis("Steering")
                self.joystick_config["throttle"] = self.map_joystick_axis("Throttle")
                self.joystick_config["brake"] = self.map_joystick_axis("Brake")

                self.joystick_config["left"] = self.map_joystick_button("Left Command")
                self.joystick_config["center"] = self.map_joystick_button("Center Command")
                self.joystick_config["right"] = self.map_joystick_button("Right Command")
                self.joystick_config["collect"] = self.map_joystick_button("Collect Data")
                self.joystick_config["autopilot"] = self.map_joystick_button("Autopilot")
                self.joystick_config["navigate"] = self.map_joystick_button("Navigate")
                self.joystick_config["train_ddpg"] = self.map_joystick_button("Train DDPG")


                self.save_joystick_mapping_to_config(self.joystick_config)

    def _draw_text(self, screen, text, color, x, y):
        font = pygame.font.Font(None, 24)
        text_surface = font.render(text, True, color)
        text_pos = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_pos)

    def map_joystick_button(self, button_name):
        """
        Maps a physical joystick button to a control button by detecting button press.

        Parameters:
            button_name (str): The name of the control button ("Left" or "Right") to map.
        Returns:
            int: The button number that has been mapped to the control button.
        """
        print_formatted("Press the joystick button you want to map to %s" % button_name)
        clock = pygame.time.Clock()
        screen = pygame.display.get_surface()
        w, h = screen.get_width(), screen.get_height()

        button = None
        running = True
        while running:
            screen.fill((0, 0, 0))
            self._draw_text(screen, "Press the joystick button you want to map to %s" % button_name, (255, 255, 255), w // 2, h // 2)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    button = event.button
                    running = False
            clock.tick(60)

        print_formatted("Button %d mapped to %s" % (button, button_name), GREEN)
        return button

    def map_joystick_axis(self, axis_name):
        """
        Maps a physical joystick axis to a control axis by detecting axis movement.

        Parameters:
            axis_name (str): The name of the control axis ("Steering" or "Throttle") to map.
        Returns:
            int: The axis number that has been mapped to the control axis.
        """
        print_formatted("Move the joystick axis you want to map to %s" % axis_name)
        clock = pygame.time.Clock()
        screen = pygame.display.get_surface()
        w, h = screen.get_width(), screen.get_height()

        axis = None
        steps = {}
        current_value = 0.0

        running = True
        while running:
            screen.fill((0, 0, 0))
            self._draw_text(screen, "Move the joystick axis you want to map to %s" % axis_name, (255, 255, 255), w // 2, h // 2)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    steps[event.axis] = steps.get(event.axis, 0) + 1
                    if steps[event.axis] > 60:
                        axis = event.axis
                        running = False
            clock.tick(60)

        print_formatted("Axis %d mapped to %s" % (axis, axis_name), GREEN)
        print_formatted(f"Move the {axis_name} axis to its minimum and maximum values, then press 'c' to continue.")

        min_val, max_val = 0.0, 0.0

        running = True
        while running:
            screen.fill((0, 0, 0))
            self._draw_text(screen, f"Move the {axis_name} axis to its minimum and maximum values, then press 'c' to continue", (255, 255, 255), w // 2, h // 2)

            self._draw_text(screen, f"Min Value: {min_val:.2f}", (255, 255, 255), w // 2 - 150, h // 2 + 50)
            self._draw_text(screen, f"Max Value: {max_val:.2f}", (255, 255, 255), w // 2 + 150, h // 2 + 50)
            pygame.display.flip()

            current_value = self.joystick.get_axis(axis)
            min_val = min(min_val, current_value)
            max_val = max(max_val, current_value)

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        running = False

        reversed = False
        running = True
        while running:
            screen.fill((0, 0, 0))
            self._draw_text(screen, f"Is the {axis_name} axis reversed? Press 'y' for yes, 'n' for no.", (255, 255, 255), w // 2, h // 2)
            self._draw_text(screen, f"Current Value: {current_value:.2f}", (255, 255, 255), w // 2, h // 2 + 50)
            pygame.display.flip()

            current_value = self.joystick.get_axis(axis)

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        reversed = True
                        running = False
                    if event.key == pygame.K_n:
                        reversed = False
                        running = False

        return {
            "mapping": axis,
            "min": min_val,
            "max": max_val,
            "reversed": reversed
        }

    def save_joystick_mapping_to_config(self, config):
        """
        Saves the current joystick axis mappings to the configuration file.
        """
        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)

        with open(self.config_file, 'w') as f:
            json.dump(config, f)

    def load_joystick_mapping(self):
        """
        Loads joystick mapping from a configuration file, if it exists.
        """
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get_joystick_control(self):
        """
        Retrieves joystick inputs based on the mapped axes for throttle and steering.

        Returns:
            tuple: A pair of (throttle, steer) values derived from joystick inputs.
        """
        pygame.event.pump()
        steer = self.normalize_axis("steer")
        throttle = self.normalize_axis("throttle")
        brake = self.normalize_axis("brake")

        throttle = max(0.0, throttle)
        brake = max(0.0, brake)

        return steer, throttle, brake

    def normalize_axis(self, name):
        """
        Normalizes the joystick axis value based on its configured min and max values.
        """
        config = self.joystick_config.get(name, {})
        value = self.joystick.get_axis(config.get("mapping"))
        min_val, max_val = config.get('min', 0), config.get('max', 1)

        if config.get('reversed', False):
            value = -value

        if name == "steer":
            value = 2 * ((value - min_val) / (max_val - min_val)) - 1
        else:
            value = (value - min_val) / (max_val - min_val)

        return round(value, 5)

    def toggle_input_method(self):
        """
        Toggles the input method between joystick and keyboard based on availability.
        """
        if self.joystick is not None:
            if self.input_method == "keyboard":
                self.input_method = "joystick"
                print_formatted("Using Joystick Input", GREEN)
            elif self.input_method == "joystick":
                self.input_method = "keyboard"
                print_formatted("Using Keyboard Input", GREEN)

    def is_inputting(self):
        """
        Checks if any input is currently being received.

        Returns:
            bool: True if input is being received, False otherwise.
        """
        steer, throttle, brake = self.get_input()
        return any([abs(steer) > 0.03, throttle > 0.03, brake > 0.03])

