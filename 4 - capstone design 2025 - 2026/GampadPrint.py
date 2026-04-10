from Quanser.q_ui import gamepadViaTarget
import time


BUTTON_ATTRS = {
	"A": "A",
	"B": "B",
	"X": "X",
	"Y": "Y",
	"LB": "LB",
	"RB": "RB",
	"Up": "up",
	"Down": "down",
	"Left": "left",
	"Right": "right",
	"Back": "back",
	"Start": "start",
}

STICK_ATTRS = {
	"Left Stick": ("LLA", "LLO"),
	"Right Stick": ("RLA", "RLO"),
}

STICK_DEADZONE = 0.4


def get_button_states(gamepad):
	states = {}
	for label, attr in BUTTON_ATTRS.items():
		if hasattr(gamepad, attr):
			states[label] = bool(getattr(gamepad, attr))
	return states


def get_stick_axis(gamepad, attr_name):
	if not hasattr(gamepad, attr_name):
		return 0.0
	return float(getattr(gamepad, attr_name))


def axis_to_cardinal(horizontal, vertical, deadzone=STICK_DEADZONE):
	horizontal_direction = ""
	vertical_direction = ""

	if horizontal >= deadzone:
		horizontal_direction = "East"
	elif horizontal <= -deadzone:
		horizontal_direction = "West"

	if vertical >= deadzone:
		vertical_direction = "North"
	elif vertical <= -deadzone:
		vertical_direction = "South"

	if vertical_direction and horizontal_direction:
		return vertical_direction + horizontal_direction
	if vertical_direction:
		return vertical_direction
	if horizontal_direction:
		return horizontal_direction
	return "Centered"


def get_stick_directions(gamepad):
	directions = {}
	for stick_name, (horizontal_attr, vertical_attr) in STICK_ATTRS.items():
		horizontal = get_stick_axis(gamepad, horizontal_attr)
		vertical = get_stick_axis(gamepad, vertical_attr)
		directions[stick_name] = axis_to_cardinal(horizontal, vertical)
	return directions


gpad = gamepadViaTarget(1)

gpad.read()
previous_button_states = get_button_states(gpad)
previous_stick_directions = get_stick_directions(gpad)

try:
	while True:
		gpad.read()
		current_button_states = get_button_states(gpad)
		current_stick_directions = get_stick_directions(gpad)
		newly_pressed = [
			name
			for name, pressed in current_button_states.items()
			if pressed and not previous_button_states.get(name, False)
		]

		if newly_pressed:
			for button_name in newly_pressed:
				print(button_name)

		for stick_name, direction in current_stick_directions.items():
			previous_direction = previous_stick_directions.get(stick_name)
			if direction != previous_direction and direction != "Centered":
				print(f"{stick_name} pointed {direction}")

		previous_button_states = current_button_states
		previous_stick_directions = current_stick_directions
		time.sleep(0.01)

except KeyboardInterrupt:
	print("User interrupted!")
finally:
	gpad.terminate()
