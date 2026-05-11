import cv2
import numpy as np
import pytesseract
import json
import difflib
"""Switch 2 Hardware code. Original layout used on Emulator, but changed things for the emulator code 'PokemonVis_TypeAd.py'. Somewhat more issues than on software"""
class PokemonAI:
    def __init__(self):
        try:
            with open("Pokemon.json", "r") as f: #Accessing data sets
                p_data = json.load(f)
                self.pokemon_db = {}
                for p in p_data:
                    #Need to deal with nidoran line female/male 
                    """Work still to deal with issues of Nidoran Female, but its evolution works"""
                    name_raw = p["name"].upper().strip()
                    if "NIDORAN" in name_raw:
                        # Map gender symbols to standard characters for better matching
                        if "♀" in name_raw:
                            clean_key = "NIDORAN F"
                        elif "♂" in name_raw:
                            clean_key = "NIDORAN M"
                        else:
                            clean_key = name_raw
                    else:
                        clean_key = p["name"].replace("♀", "").replace("♂", "").upper().strip()
                    self.pokemon_db[clean_key] = p
            with open("Type_Matching.json", "r") as f:
                t_data = json.load(f)
                self.type_chart = {t["Type"].upper(): t for t in t_data}
        except FileNotFoundError as e:
            print(f"File error: {e}")
            self.pokemon_db = {}
            self.type_chart = {}
        
        self.front_path = "sprites/front"
        
        # --- PRESERVED POSITIONS ---
        self.opp_roi = (280, 105, 835, 265)
        self.my_roi = (1080, 482, 1720, 605)
        self.move_menu_roi = (180, 745, 1750, 1060)
        self.hp_offsets = {
            "OPP": (240, 118, 550, 125),
            "MY":  (231, 122, 540, 124)
        }
        self.move_slots = {
            "Move1": (50, 50, 500, 150),
            "Move2": (510, 50, 970, 150),
            "Move3": (50, 160, 500, 250),
            "Move4": (510, 160, 970, 250),
            "Type": (1080, 160, 1500, 250)
        }
        
        self.name_offsets = {
            "OPP": (19, 30, 360, 90), 
            "MY":  (15, 29, 400, 95)
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
            "Overworld": (0, 165, 255),
            "SELECT_MOVE": (255, 255, 255),
            "Move1": (150,150,150),
            "Move2": (150,150,150),
            "Move3": (150,150,150),
            "Move4": (150,150,150),
            "Type": (150,150,150)
        }

    def moveAdvantage(self, current_move_type, opponent_name):
        opponent_name = opponent_name.upper()
        current_move_type = current_move_type.upper()
        if opponent_name not in self.pokemon_db or current_move_type not in self.type_chart:
            return 1.0 
        target_types = self.pokemon_db[opponent_name]["type"]
        print(target_types)
        type_data = self.type_chart[current_move_type]
        final_mult = 1.0
        for t in target_types:
            t_upper = t.upper()
            if t_upper in [w.upper() for w in type_data.get("Weak", [])]: final_mult *= 0.5
            elif t_upper in [s.upper() for s in type_data.get("Strengths", [])]: final_mult *= 2.0
            elif t_upper in [i.upper() for i in type_data.get("Immune", [])]: final_mult *= 0.0
        return final_mult
    
    def get_sprite(self, name, is_opponent=True):
        name_upper = name.upper()
        if name_upper not in self.pokemon_db: return None
        poke_id = self.pokemon_db[name_upper]["id"]
        path = f"{self.front_path}/{poke_id}.png" 
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

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
        beige_mask = cv2.inRange(hsv, np.array([25, 5, 215]), np.array([35, 45, 255]))
        return np.count_nonzero(beige_mask) > 3000

    def get_name_via_ocr(self, plate_roi, is_opponent=True):
        key = "OPP" if is_opponent else "MY"
        x1, y1, x2, y2 = self.name_offsets[key] 
        crop = plate_roi[y1:y2, x1:x2]
        if crop.size == 0: return "UNKNOWN"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
        config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ' 
        raw = pytesseract.image_to_string(thresh, config=config).upper().strip()
        clean = "".join(c for c in raw if c.isalpha())
        if len(clean) < 3: return "UNKNOWN"
        matches = difflib.get_close_matches(clean, self.pokemon_db.keys(), n=1, cutoff=0.45)
        return matches[0] if matches else "UNKNOWN"

    def draw_hp_status(self, dashboard):
        if self.current_state not in ["BATTLE", "SELECT_MOVE"]: return
        def gba_col(p): return (0, 255, 0) if p > 50 else (0, 255, 255) if p > 20 else (0, 0, 255) 
        for k, y, is_opp in [("OPP", 120, True), ("MY", 260, False)]:
            name = self.detected_names[k]
            hp = int(self.hp_tracker[k])
            sprite = self.get_sprite(name, is_opp)
            cv2.putText(dashboard, f"{name} {hp}%", (20, y), 1, 1.4, (255, 255, 255), 2)
            if sprite is not None: self.overlay_sprite(dashboard, sprite, 280, y-70)
            cv2.rectangle(dashboard, (20, y+50), (20 + int(self.hp_tracker[k] * 3), y+60), gba_col(self.hp_tracker[k]), -1)

    def get_hp_percentage(self, frame, plate_coords, is_opponent=True):
        px1, py1, px2, py2 = plate_coords 
        plate_roi = frame[py1:py2, px1:px2]
        key = "OPP" if is_opponent else "MY"
        hx1, hy1, hx2, hy2 = self.hp_offsets[key]
        
        hp_bar_crop = plate_roi[hy1:hy2, hx1:hx2]
        if hp_bar_crop.size == 0: return None

        # SWITCH 2 FIX: Look for light intensity rather than specific Hue
        gray = cv2.cvtColor(hp_bar_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        
        w = thresh.shape[1]
        filled = 0
        for x in range(w):
            if np.any(thresh[:, x] > 0): filled += 1
            elif x > 5: break
        return (filled / w) * 100

    def highlight_hp_slots(self, frame, coords, is_opponent=True):
        px1, py1, px2, py2 = coords
        key = "OPP" if is_opponent else "MY"
        hx1, hy1, hx2, hy2 = self.hp_offsets[key]
        cv2.rectangle(frame, (px1 + hx1, py1 + hy1), (px1 + hx2, py1 + hy2), (0, 0, 255), 2) 

    def process_frame(self, frame):
        self.frame_count += 1
        frame = cv2.resize(frame, (1920, 1080))
        display_game, dashboard = frame.copy(), np.zeros((1080, 400, 3), dtype=np.uint8)

        # Menu Checks
        # --- REVISED SUMMARY DETECTION ---
        # Targets only the thick blue header bar found on the Summary info page
        s_y1, s_y2, s_x1, s_x2 = 10, 110, 150, 1750
        hsv_s = cv2.cvtColor(frame[s_y1:s_y2, s_x1:s_x2], cv2.COLOR_BGR2HSV)

        # Narrow Hue to [95-115] to ignore teal/green found in the Party menu
        # Require high Saturation and Value to target vivid UI blue
        summary_mask = cv2.inRange(hsv_s, np.array([95, 150, 150]), np.array([115, 255, 255]))

        # Requirement increased from 12,000 to 80,000 pixels
        # This ensures it only triggers when the massive header bar is present
        is_summary = (np.count_nonzero(summary_mask) > 40000)

        p_y1, p_y2, p_x1, p_x2 = 870, 1050, 1370, 1750
        py_mask = cv2.inRange(cv2.cvtColor(frame[p_y1:p_y2, p_x1:p_x2], cv2.COLOR_BGR2HSV), np.array([80, 50, 150]), np.array([90, 255, 255]))
        is_party = (np.count_nonzero(py_mask) > 10000)

        b_y1, b_y2, b_x1, b_x2 = 750, 1060, 150, 1750
        blue_mask = cv2.inRange(cv2.cvtColor(frame[b_y1:b_y2, b_x1:b_x2], cv2.COLOR_BGR2HSV), np.array([95, 240, 200]), np.array([120, 254, 250]))
        is_bag = (np.count_nonzero(blue_mask) > 3000) and not is_party

        # Move Menu Detection
        m_x1, m_y1, m_x2, m_y2 = self.move_menu_roi
        move_roi_img = frame[m_y1:m_y2, m_x1:m_x2]
        m_hsv = cv2.cvtColor(move_roi_img, cv2.COLOR_BGR2HSV)
        move_bg_mask = cv2.inRange(m_hsv, np.array([0, 0, 240]), np.array([179, 15, 255]))
        is_move_menu = (np.count_nonzero(move_bg_mask) > 290000)

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
            if is_move_menu:
                self.current_state = "SELECT_MOVE"
                tx1, ty1, tx2, ty2 = self.move_slots["Type"]
                type_roi = move_roi_img[ty1:ty2, tx1:tx2]
                # ADD THIS LOOP TO DRAW THE BOXES:
                for move_name, (mx1, my1, mx2, my2) in self.move_slots.items():
                    # Draw boxes relative to the move_menu_roi position
                    start_point = (self.move_menu_roi[0] + mx1, self.move_menu_roi[1] + my1)
                    end_point = (self.move_menu_roi[0] + mx2, self.move_menu_roi[1] + my2)
                    cv2.rectangle(display_game, start_point, end_point, self.colors.get(move_name, (255, 255, 255)), 2)
                gray_t = cv2.cvtColor(type_roi, cv2.COLOR_BGR2GRAY)
                _, thresh_t = cv2.threshold(gray_t, 150, 255, cv2.THRESH_BINARY_INV)
                type_raw = pytesseract.image_to_string(thresh_t, config='--psm 8').strip().upper()
                
                opp_name = self.detected_names["OPP"]
                advantage = 1.0
                if type_raw and opp_name != "UNKNOWN":
                    matches = difflib.get_close_matches(type_raw, self.type_chart.keys(), n=1, cutoff=0.55)
                    if matches: advantage = self.moveAdvantage(matches[0], opp_name)

                adv_text = f"EFFECTIVENESS: {advantage}x"
                adv_col = (0, 255, 0) if advantage > 1 else (0, 0, 255) if advantage < 1 else (255, 255, 255) #Can add in another color for 4x effective just need to get to a pokemon which has both typings a move is strong against.
                cv2.putText(dashboard, adv_text, (20, 450), 1, 1.5, adv_col, 2)
                cv2.rectangle(display_game, (m_x1, m_y1), (m_x2, m_y2), self.colors["SELECT_MOVE"], 3)
            else:
                self.current_state = "BATTLE"
            
            self.state_buffer = self.buffer_max
            for k, roi, is_opp in [("OPP", self.opp_roi, True), ("MY", self.my_roi, False)]:
                active = opp_active if is_opp else my_active
                if active:
                    x1, y1, x2, y2 = roi
                    if self.frame_count % 60 == 0:
                        res = self.get_name_via_ocr(frame[y1:y2, x1:x2], is_opp)
                        if res != "UNKNOWN": self.detected_names[k] = res
                    val = self.get_hp_percentage(frame, roi, is_opp)
                    if val is not None: self.hp_tracker[k] = val
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

bot = PokemonAI()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    out = bot.process_frame(frame)
    cv2.imshow("Pokemon Move Effectiveness", cv2.resize(out, (1280, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()