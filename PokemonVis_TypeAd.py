import cv2
import numpy as np

class PokemonAI:
    def __init__(self):
        # We no longer need a list of anchors for state switching
        # The "State" is now just a check: "Is there HP color in the box?"
        pass

    def get_hp_data(self, frame, x1, y1, x2, y2):
        """Looks for HP bar colors (Green/Yellow/Red) in a specific box."""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return 0, 0
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Mask for Green (Full), Yellow (Half), and Red (Low) HP bars
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([100, 255, 255]))
        
        pixel_count = np.count_nonzero(mask)
        percentage = (pixel_count / mask.size) * 100
        return percentage, pixel_count

    def process_frame(self, frame):
        h, w, _ = frame.shape
        display_game = frame.copy()
        dashboard = np.zeros((h, 400, 3), dtype=np.uint8)

        # 1. Define your Pokemon Panel Coordinates
        opp_roi = (100, 125, 700, 275)   # Pidgey Area
        my_roi = (915, 500, 1520, 720)   # Charmander Area

        # 2. Check for HP pixels in those areas
        opp_hp, opp_pixels = self.get_hp_data(frame, *opp_roi)
        my_hp, my_pixels = self.get_hp_data(frame, *my_roi)

        # 3. ONLY DRAW IF PIXELS ARE FOUND (Threshold: 50 pixels)
        # This acts as your "Overworld Filter"
        if opp_pixels > 50 or my_pixels > 50:
            # Draw the Blue Squares
            cv2.rectangle(display_game, (opp_roi[0], opp_roi[1]), (opp_roi[2], opp_roi[3]), (255, 0, 0), 3)
            cv2.rectangle(display_game, (my_roi[0], my_roi[1]), (my_roi[2], my_roi[3]), (255, 0, 0), 3)

            # Show Data on Dashboard
            cv2.putText(dashboard, "BATTLE DETECTED", (20, 50), 1, 1.8, (0, 255, 0), 2)
            cv2.putText(dashboard, f"OPP HP: {int(opp_hp)}%", (20, 120), 1, 1.2, (255, 255, 255), 1)
            cv2.putText(dashboard, f"MY HP: {int(my_hp)}%", (20, 160), 1, 1.2, (255, 255, 255), 1)
        else:
            cv2.putText(dashboard, "WALKING / NO HP", (20, 50), 1, 1.8, (255, 255, 0), 2)

        return np.hstack((display_game, dashboard))

# --- EXECUTION ---
bot = PokemonAI()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    output = bot.process_frame(frame)
    cv2.imshow("Pokemon AI - Reactive Mode", output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()