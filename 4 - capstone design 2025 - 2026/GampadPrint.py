from Quanser.q_ui import gamepadViaTarget
import time


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


def print_snapshot(snapshot):
	print("Available gamepad fields:")
	for attr_name in sorted(snapshot):
		print(f"  {attr_name}: {snapshot[attr_name]}")


def values_differ(previous_value, current_value):
	if isinstance(previous_value, float) or isinstance(current_value, float):
		return abs(float(previous_value) - float(current_value)) > 0.05
	return previous_value != current_value


gpad = gamepadViaTarget(1)

gpad.read()
previous_snapshot = get_gamepad_snapshot(gpad)

print("Press a button")
print_snapshot(previous_snapshot)

try:
	while True:
		gpad.read()
		current_snapshot = get_gamepad_snapshot(gpad)

		for attr_name in sorted(current_snapshot):
			previous_value = previous_snapshot.get(attr_name)
			current_value = current_snapshot[attr_name]
			if previous_value is None or values_differ(previous_value, current_value):
				print(f"{attr_name}: {previous_value} -> {current_value}")

		previous_snapshot = current_snapshot
		time.sleep(0.01)

except KeyboardInterrupt:
	print("User interrupted!")
finally:
	gpad.terminate()
