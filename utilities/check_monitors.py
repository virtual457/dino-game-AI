import pyautogui

# Get screen info
screen_width, screen_height = pyautogui.size()

print("Screen Info:")
print(f"Total screen width: {screen_width}")
print(f"Total screen height: {screen_height}")
print()

# Try to determine monitor layout
# If you have 3 monitors side by side, total width would be ~7680 (2560*3)
# Or they might be arranged differently

print("Your setup:")
if screen_width > 7000:
    print("Looks like 3 monitors horizontal")
    print(f"Monitor 1: 0 to 2560")
    print(f"Monitor 2: 2560 to 5120")
    print(f"Monitor 3: 5120 to 7680")
elif screen_width > 5000:
    print("Looks like 2 monitors + partial")
else:
    print("Single monitor or vertical arrangement")

print(f"\nTotal virtual screen: {screen_width} x {screen_height}")
