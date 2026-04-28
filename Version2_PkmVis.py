import cv2
import numpy as np
import pytesseract
import json
import difflib
import os

class PokemonAI:
    def __init__(self):
        try:
            with open("Pokemon.json", "r") as f:
                p_data = json.load(f)
                self.pokemon_db = {}
                for p in p_data:
                    clean_key = p["name"].replace("♀", "").replace("♂", "").upper().strip()
                    self.pokemon_db[clean_key] = p
            with open("Type_Matching.json", "r") as f:
                t_data = json.load(f)
                self.type_chart = {t["Type"].upper(): t for t in t_data}
        except FileNotFoundError as e:
            print(f"File error: {e}")
            self.pokemon_db = {}
            self.type_chart = {}

        self.front_path = "sprites/font"
        self.back_path = "sprites/back"
        
        # Battle Regions (1920x1080)
        self.opp_roi = (100, 125, 700, 275)
        self.my_roi = (915, 500, 1520, 720)
        
        # Name Offsets relative to the plate
        self.name_offsets = {
            "OPP": (15, 12, 350, 88), 
            "MY":  (15, 32, 380, 102)
        }
        
        self.detected_names = {"OPP": "UNKNOWN", "MY": "UNKNOWN"}
        self.current_state = "Overworld"
        self.state_buffer = 0
        self.buffer_max = 12
        self.hp_tracker = {"OPP": 100.0, "MY": 100.0}
        self.frame_count = 0
        self.colors = {
            "BATTLE": (0, 255, 0),
            "Bag": (255, 0, 0),
            "Summary": (0, 255, 255),
            "Party": (255, 0, 255),
            "Overworld": (0, 165, 255)
        }

    def get_sprite(self, name, is_opponent=True):
        name_upper = name.upper()
        if name_upper not in self.pokemon_db: return None
        poke_id = self.pokemon_db[name_upper]["id"]
        folder = self.front_path if is_opponent else self.back_path
        path = f"{folder}/{poke_id}.png"
        return cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None

    def overlay_sprite(self, dashboard, sprite, x, y, size=(90, 90)):
        if sprite is None: return
        sprite = cv2.resize(sprite, size)
        if sprite.shape[2] == 4:
            alpha = sprite[:, :, 3] / 255.0
            for c in range(3):
                dashboard[y:y+size[1], x:x+size[0], c] = \
                    (sprite[:, :, c] * alpha) + (dashboard[y:y+size[1], x:x+size[0], c] * (1.0 - alpha))
        else:
            dashboard[y:y+size[1], x:x+size[0]] = sprite

    def is_plate_present(self, frame, coords):
        x1, y1, x2, y2 = coords
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        beige_mask = cv2.inRange(hsv, np.array([10, 20, 150]), np.array([30, 80, 255]))
        return np.count_nonzero(beige_mask) > 3500

    # --- RESTORED ROBUST NAME LOGIC ---
    # --- IMPROVED ROBUST NAME LOGIC ---
    def get_name_via_ocr(self, plate_roi, is_opponent=True):
        key = "OPP" if is_opponent else "MY"
        x1, y1, x2, y2 = self.name_offsets[key]
        crop = plate_roi[y1:y2, x1:x2]
        if crop.size == 0: return "UNKNOWN"
        
        # NEAREST interpolation is faster and better for GBA pixel font edges
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        raw = pytesseract.image_to_string(thresh, config=config).upper().strip()
        
        clean = "".join(c for c in raw if c.isalpha())
        if len(clean) < 3: return "UNKNOWN"
        
        matches = difflib.get_close_matches(clean, self.pokemon_db.keys(), n=1, cutoff=0.65)
        return matches[0] if matches else "UNKNOWN"

    def draw_hp_status(self, dashboard):
        if self.current_state != "BATTLE": return
        def gba_col(p): return (0, 255, 0) if p > 50 else (0, 255, 255) if p > 20 else (0, 0, 255)
        
        for k, y, is_opp in [("OPP", 120, True), ("MY", 260, False)]:
            name = self.detected_names[k]
            hp = int(self.hp_tracker[k])
            
            # Retrieve the sprite using the ID from your JSON
            sprite = self.get_sprite(name, is_opp)
            
            # Display logic
            cv2.putText(dashboard, f"{name} {hp}%", (20, y), 1, 1.4, (255, 255, 255), 2)
            if sprite is not None:
                self.overlay_sprite(dashboard, sprite, 280, y-70)
            
            # HP Bar rendering
            cv2.rectangle(dashboard, (20, y+50), (320, y+60), (40, 40, 40), -1)
            cv2.rectangle(dashboard, (20, y+50), (20 + int(self.hp_tracker[k] * 3), y+60), gba_col(self.hp_tracker[k]), -1)

    def get_hp_percentage(self, frame, plate_coords, is_opponent=True):
        px1, py1, px2, py2 = plate_coords
        plate_roi = frame[py1:py2, px1:px2]
        hx1, hy1, hx2, hy2 = (244, 108, 560, 114) if is_opponent else (237, 116, 553, 122)
        hp_bar_crop = plate_roi[hy1:hy2, hx1:hx2]
        if hp_bar_crop.size == 0: return None
        hsv = cv2.cvtColor(hp_bar_crop, cv2.COLOR_BGR2HSV)
        w, filled = hsv.shape[1], 0
        for x in range(w):
            if np.any((hsv[:, x][:, 1] > 70) & (hsv[:, x][:, 2] > 60)): filled += 1
            elif x > 5: break
        return (filled / w) * 100

    def highlight_hp_slots(self, frame, coords, is_opponent=True):
        px1, py1, px2, py2 = coords
        hx1, hy1, hx2, hy2 = (244, 108, 560, 114) if is_opponent else (237, 116, 553, 122)
        cv2.rectangle(frame, (px1 + hx1, py1 + hy1), (px1 + hx2, py1 + hy2), (0, 0, 255), 2)

    def process_frame(self, frame):
        self.frame_count += 1
        frame = cv2.resize(frame, (1920, 1080))
        display_game, dashboard = frame.copy(), np.zeros((1080, 400, 3), dtype=np.uint8)

        # --- MENU DETECTION ---
        s_y1, s_y2, s_x1, s_x2 = 10, 110, 0, 1600
        hsv_s = cv2.cvtColor(frame[s_y1:s_y2, s_x1:s_x2], cv2.COLOR_BGR2HSV)
        is_summary = (np.count_nonzero(cv2.inRange(hsv_s, np.array([90, 100, 50]), np.array([110, 255, 220]))) > 12000) and \
                     (np.count_nonzero(cv2.inRange(hsv_s, np.array([0, 0, 245]), np.array([180, 10, 255]))) > 600)

        p_y1, p_y2, p_x1, p_x2 = 880, 1060, 0, 1600
        w_mask = cv2.inRange(cv2.cvtColor(frame[p_y1:p_y2, p_x1:p_x2], cv2.COLOR_BGR2HSV), np.array([0, 0, 230]), np.array([180, 25, 255]))
        bg_mask = cv2.inRange(cv2.cvtColor(frame[200:400, 100:400], cv2.COLOR_BGR2HSV), np.array([70, 50, 50]), np.array([100, 255, 255]))
        is_party = (np.count_nonzero(w_mask) > 60000) and (np.count_nonzero(bg_mask) > 10000)

        b_y1, b_y2, b_x1, b_x2 = 750, 1060, 0, 1600
        blue_mask = cv2.inRange(cv2.cvtColor(frame[b_y1:b_y2, b_x1:b_x2], cv2.COLOR_BGR2HSV), np.array([100, 150, 100]), np.array([110, 255, 255]))
        is_bag = (np.count_nonzero(blue_mask) > 35000) and not is_party

        # --- BATTLE LOGIC ---
        opp_active, my_active = self.is_plate_present(frame, self.opp_roi), self.is_plate_present(frame, self.my_roi)

        if is_summary:
            self.current_state = "Summary"
            cv2.rectangle(display_game, (s_x1, s_y1), (s_x2, s_y2), self.colors["Summary"], 3)
        elif is_party:
            self.current_state = "Party"
            cv2.rectangle(display_game, (p_x1, p_y1), (p_x2, p_y2), self.colors["Party"], 3)
        elif is_bag:
            self.current_state = "Bag"
            cv2.rectangle(display_game, (b_x1, b_y1), (b_x2, b_y2), self.colors["Bag"], 3)
        elif opp_active or my_active:
            self.current_state, self.state_buffer = "BATTLE", self.buffer_max
            for k, roi, is_opp in [("OPP", self.opp_roi, True), ("MY", self.my_roi, False)]:
                active = opp_active if is_opp else my_active
                
                if active:
                    x1, y1, x2, y2 = roi
                    
                    # --- INSTANT POLLING LOGIC ---
                    # If UNKNOWN: check every 2 frames (near-instant)
                    # If KNOWN: check every 60 frames (background refresh for swaps)
                    check_rate = 2 if self.detected_names[k] == "UNKNOWN" else 60
                    
                    if self.frame_count % check_rate == 0:
                        res = self.get_name_via_ocr(frame[y1:y2, x1:x2], is_opp)
                        if res != "UNKNOWN": 
                            self.detected_names[k] = res

                    # HP tracking (keep at 1.0 alpha for opponent to see damage instantly)
                    val = self.get_hp_percentage(frame, roi, is_opp)
                    if val is not None:
                        alpha = 1.0 if is_opp else 0.20
                        self.hp_tracker[k] = (alpha * val) + ((1 - alpha) * self.hp_tracker[k])

                    # Visual Overlays
                    nx1, ny1, nx2, ny2 = self.name_offsets[k]
                    cv2.rectangle(display_game, (x1+nx1, y1+ny1), (x1+nx2, y1+ny2), (255, 255, 0), 2)
                    self.highlight_hp_slots(display_game, roi, is_opp)
        else:
            if self.state_buffer > 0: 
                self.state_buffer, self.current_state = self.state_buffer - 1, "BATTLE"
            else:
                self.current_state = "Overworld"
                self.detected_names = {"OPP": "UNKNOWN", "MY": "UNKNOWN"}

        cv2.putText(dashboard, f"STATE: {self.current_state}", (20, 60), 1, 2, self.colors.get(self.current_state, (255,255,255)), 2)
        self.draw_hp_status(dashboard)
        return np.hstack((display_game, dashboard))

# Execution loop
bot = PokemonAI()
cap = cv2.VideoCapture(0)
is_paused = False
while cap.isOpened():
    if not is_paused:
        ret, frame = cap.read()
        if not ret: break
        out = bot.process_frame(frame)
        cv2.imshow("Pokemon AI", cv2.resize(out, (1280, 600)))
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): is_paused = not is_paused
    if key == ord('q'): break
cap.release()
cv2.destroyAllWindows()