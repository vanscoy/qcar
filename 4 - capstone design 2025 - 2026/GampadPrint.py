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
}


STICK_ATTRS = {
	"Left Stick": ("LLO", "LLA"),
	"Right Stick": ("RLO", "RLA"),
}


TRIGGER_ATTRS = {
	"LT": "LT",
	"RT": "RT",
}


STICK_DEADZONE = 0.4
TRIGGER_DEADZONE = 0.2
CONTROLLER_INDICES = (1, 2, 3, 4)
WAIT_MESSAGE_INTERVAL = 3.0


def is_trackable_value(value):
	return isinstance(value, (bool, int, float, str))


def get_gamepad_snapshot(gamepad):
	snapshot = {}
	for attr_name in dir(gamepad):
		if attr_name.startswith("_"):
			continue
		try:
			value = getattr(gamepad, attr_name)
		except Exception:
			continue
		if callable(value) or not is_trackable_value(value):
			continue
		snapshot[attr_name] = value
	return snapshot


def values_differ(previous_value, current_value):
	if isinstance(previous_value, float) or isinstance(current_value, float):
		return abs(float(previous_value) - float(current_value)) > 0.05
	return previous_value != current_value


def get_button_states(gamepad):
	states = {}
	for label, attr in BUTTON_ATTRS.items():
		states[label] = bool(getattr(gamepad, attr, 0))
	return states


def get_trigger_states(gamepad):
	states = {}
	for label, attr in TRIGGER_ATTRS.items():
		states[label] = float(getattr(gamepad, attr, 0.0)) >= TRIGGER_DEADZONE
	return states


def get_axis(gamepad, attr_name):
	return float(getattr(gamepad, attr_name, 0.0))


def axis_to_direction(horizontal, vertical):
	horizontal_direction = ""
	vertical_direction = ""

	if horizontal >= STICK_DEADZONE:
		horizontal_direction = "East"
	elif horizontal <= -STICK_DEADZONE:
		horizontal_direction = "West"

	if vertical >= STICK_DEADZONE:
		vertical_direction = "North"
	elif vertical <= -STICK_DEADZONE:
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
		horizontal = get_axis(gamepad, horizontal_attr)
		vertical = get_axis(gamepad, vertical_attr)
		directions[stick_name] = axis_to_direction(horizontal, vertical)
	return directions


def connect_gamepads(indices):
	controllers = []
	for index in indices:
		try:
			gamepad = gamepadViaTarget(index)
			gamepad.read()
		except Exception as exc:
			print(f"Controller {index}: unavailable ({exc})")
			continue

		controllers.append(
			{
				"index": index,
				"gamepad": gamepad,
				"previous_button_states": get_button_states(gamepad),
				"previous_trigger_states": get_trigger_states(gamepad),
				"previous_stick_directions": get_stick_directions(gamepad),
				"previous_snapshot": get_gamepad_snapshot(gamepad),
			}
		)
		print(f"Controller {index}: listening")
	return controllers


controllers = connect_gamepads(CONTROLLER_INDICES)

known_attrs = set(BUTTON_ATTRS.values())
known_attrs.update(TRIGGER_ATTRS.values())
for horizontal_attr, vertical_attr in STICK_ATTRS.values():
	known_attrs.add(horizontal_attr)
	known_attrs.add(vertical_attr)

activity_detected = False
last_wait_message_time = time.time()

print("Press a button")
print("Waiting for controller input...")

try:
	while True:
		loop_activity_detected = False
		for controller in controllers:
			gamepad = controller["gamepad"]
			gamepad.read()
			current_button_states = get_button_states(gamepad)
			current_trigger_states = get_trigger_states(gamepad)
			current_stick_directions = get_stick_directions(gamepad)
			current_snapshot = get_gamepad_snapshot(gamepad)
			controller_activity_detected = False

			for button_name, is_pressed in current_button_states.items():
				if is_pressed and not controller["previous_button_states"].get(button_name, False):
					print(f"[Controller {controller['index']}] {button_name}")
					controller_activity_detected = True

			for trigger_name, is_pressed in current_trigger_states.items():
				if is_pressed and not controller["previous_trigger_states"].get(trigger_name, False):
					print(f"[Controller {controller['index']}] {trigger_name}")
					controller_activity_detected = True

			for stick_name, direction in current_stick_directions.items():
				if direction != controller["previous_stick_directions"].get(stick_name) and direction != "Centered":
					print(f"[Controller {controller['index']}] {stick_name} pointed {direction}")
					controller_activity_detected = True

			for attr_name in sorted(current_snapshot):
				previous_value = controller["previous_snapshot"].get(attr_name)
				current_value = current_snapshot[attr_name]
				if previous_value is None:
					continue
				if not values_differ(previous_value, current_value):
					continue
				if not activity_detected:
					print(f"Controller signal detected on controller {controller['index']}")
					activity_detected = True
				if attr_name not in known_attrs:
					print(
						f"[Controller {controller['index']}] Unidentified input: {attr_name}: {previous_value} -> {current_value}"
					)
					controller_activity_detected = True

			if controller_activity_detected and not activity_detected:
				print(f"Controller signal detected on controller {controller['index']}")
				activity_detected = True
			if controller_activity_detected:
				loop_activity_detected = True

			controller["previous_button_states"] = current_button_states
			controller["previous_trigger_states"] = current_trigger_states
			controller["previous_stick_directions"] = current_stick_directions
			controller["previous_snapshot"] = current_snapshot

		if not loop_activity_detected and time.time() - last_wait_message_time >= WAIT_MESSAGE_INTERVAL:
			if controllers:
				controller_list = ", ".join(str(controller["index"]) for controller in controllers)
				print(f"No controller input detected yet on controller indices: {controller_list}")
			else:
				print("No controller interfaces are available")
			last_wait_message_time = time.time()
		elif loop_activity_detected:
			last_wait_message_time = time.time()

		time.sleep(0.01)

except KeyboardInterrupt:
	print("User interrupted!")
finally:
	for controller in controllers:
		controller["gamepad"].terminate()
