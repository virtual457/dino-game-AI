"""
System-Level Key Monitor
Captures SPACE and DOWN keys at OS level (works even when window not focused)
Displays visual feedback with green indicators
"""

import tkinter as tk
from pynput import keyboard
import threading


class KeyMonitor:
    """Visual monitor for SPACE and DOWN key presses"""
    
    def __init__(self):
        # Key states
        self.up_pressed = False  # Changed from space_pressed
        self.down_pressed = False
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Key Monitor - System Level")
        self.root.geometry("300x200")
        self.root.attributes('-topmost', True)  # Always on top
        
        # Style
        bg_color = "#1e1e1e"
        self.root.configure(bg=bg_color)
        
        # Title
        title = tk.Label(
            self.root,
            text="Key Monitor (System-Level)",
            font=("Arial", 14, "bold"),
            bg=bg_color,
            fg="white"
        )
        title.pack(pady=10)
        
        # UP key indicator (for jump)
        self.up_label = tk.Label(
            self.root,
            text="UP ARROW (JUMP)",
            font=("Arial", 16, "bold"),
            bg="gray",
            fg="white",
            width=20,
            height=2
        )
        self.up_label.pack(pady=10)
        
        # DOWN key indicator
        self.down_label = tk.Label(
            self.root,
            text="DOWN (DUCK)",
            font=("Arial", 16, "bold"),
            bg="gray",
            fg="white",
            width=20,
            height=2
        )
        self.down_label.pack(pady=10)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="Monitoring... (Press ESC to exit)",
            font=("Arial", 9),
            bg=bg_color,
            fg="#888888"
        )
        self.status_label.pack(pady=5)
        
        # Start keyboard listener in separate thread
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        
        print("[KEY MONITOR] Started!")
        print("[INFO] This captures keys at SYSTEM LEVEL (works everywhere)")
        print("[INFO] Press ESC to exit")
        print("[INFO] Window will stay on top\n")
    
    def on_press(self, key):
        """Handle key press - OS level!"""
        try:
            # Check for UP arrow
            if key == keyboard.Key.up:
                if not self.up_pressed:
                    self.up_pressed = True
                    self.update_display()
                    print("[UP] Pressed (JUMP)")
            
            # Check for DOWN arrow
            elif key == keyboard.Key.down:
                if not self.down_pressed:
                    self.down_pressed = True
                    self.update_display()
                    print("[DOWN] Pressed (DUCK)")
            
            # ESC to exit
            elif key == keyboard.Key.esc:
                print("\n[EXIT] ESC pressed, stopping...")
                self.stop()
        
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release - OS level!"""
        try:
            # Check for UP release
            if key == keyboard.Key.up:
                if self.up_pressed:
                    self.up_pressed = False
                    self.update_display()
                    print("[UP] Released")
            
            # Check for DOWN release
            elif key == keyboard.Key.down:
                if self.down_pressed:
                    self.down_pressed = False
                    self.update_display()
                    print("[DOWN] Released")
        
        except AttributeError:
            pass
    
    def update_display(self):
        """Update the visual indicators"""
        # Update UP indicator
        if self.up_pressed:
            self.up_label.configure(bg="#00ff00", fg="black")  # Bright green
        else:
            self.up_label.configure(bg="gray", fg="white")
        
        # Update DOWN indicator
        if self.down_pressed:
            self.down_label.configure(bg="#00ff00", fg="black")  # Bright green
        else:
            self.down_label.configure(bg="gray", fg="white")
    
    def stop(self):
        """Stop the monitor"""
        self.listener.stop()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run the GUI main loop"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted")
            self.stop()


def main():
    """Main entry point"""
    print("="*70)
    print(" " * 20 + "SYSTEM-LEVEL KEY MONITOR")
    print("="*70)
    print("\nCaptures:")
    print("  • UP arrow key (for jump)")
    print("  • DOWN arrow key (for duck)")
    print("\nFeatures:")
    print("  • Works at OS/system level")
    print("  • Captures keys even when Chrome not focused")
    print("  • Green indicator when key pressed")
    print("  • Always-on-top window")
    print("\nControls:")
    print("  • Press ESC to exit")
    print("="*70 + "\n")
    
    monitor = KeyMonitor()
    monitor.run()
    
    print("\n[COMPLETE] Key monitor stopped")


if __name__ == "__main__":
    main()
