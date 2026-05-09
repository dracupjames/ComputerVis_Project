import cv2
import numpy as np
import pytesseract
import json
import difflib
#import os

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
        
        self.front_path = "sprites/front" #location of front sprites for dashboard of the pokemon
        
        # Battle Regions (1920x1080) Have to change for other similar scripts for different recordings/live runs
        self.opp_roi = (111, 120, 702, 275) # (x1,y1,x2,y2)
        self.my_roi = (915, 486, 1520, 716)
        
        # Move Menu Regions
        self.move_menu_roi = (20,750,1600,1060)

        # Summary Region
        self.summary_roi = (10, 110, 0, 1600)

        # Party Region
        self.party_roi = (870, 1030, 1190, 1559)

        # Bag Region
        self.bag_roi = (750, 1060, 0, 1600)
        
        self.move_slots = {
            "Move1": (50, 50, 500, 150),
            "Move2": (510, 50, 970, 150),
            "Move3": (50, 160, 500, 250),
            "Move4": (510, 160, 970, 250),
            "Type": (1080, 160, 1500, 250)
        }
        
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
        self.colors = { # Different colors for the detection boxes
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
        # 1. Clean inputs and check if we have data
        opponent_name = opponent_name.upper()
        current_move_type = current_move_type.upper()
        
        if opponent_name not in self.pokemon_db or current_move_type not in self.type_chart:
            return 1.0 # Neutral if unknown

        # 2. Get opponent's types (e.g., ["FIRE", "FLYING"])
        target_types = self.pokemon_db[opponent_name]["type"]
        
        # 3. Calculate the effectiveness with hardcoded multipliers here based
        # on if the Moves type will be weak, strong, or immune to the opponent
        type_data = self.type_chart[current_move_type]
        
        final_mult = 1.0
        
        for t in target_types:
            t_upper = t.upper()
            # Logic depends on your JSON structure:
            if t_upper in [w.upper() for w in type_data.get("Weak", [])]:
                final_mult *= 0.5
            elif t_upper in [s.upper() for s in type_data.get("Strengths", [])]:
                final_mult *= 2.0
            elif t_upper in [i.upper() for i in type_data.get("Immune", [])]:
                final_mult *= 0.0
                
        return final_mult
    
    def get_sprite(self, name, is_opponent=True): #Retrieve the front sprite of the pokemon to appear next to healthbar on dashboard
        name_upper = name.upper()
        if name_upper not in self.pokemon_db: return None
        poke_id = self.pokemon_db[name_upper]["id"]
        folder = self.front_path
        path = f"{folder}/{poke_id}.png" #Each sprite is saved as a number matching the dex number in FireRed
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def overlay_sprite(self, dashboard, sprite, x, y, size=(90, 90)):
        if sprite is None: return
        sprite = cv2.resize(sprite, size)
        if sprite.shape[2] == 4:
            alpha = sprite[:, :, 3] / 255.0 #alpha blending with 3 being the channel for transparency so we remove the background of the pokemon sprite
            for c in range(3):
                dashboard[y:y+size[1], x:x+size[0], c] = \
                    (sprite[:, :, c] * alpha) + (dashboard[y:y+size[1], x:x+size[0], c] * (1.0 - alpha)) #Alpha blending
        else:
            dashboard[y:y+size[1], x:x+size[0]] = sprite

    def is_plate_present(self, frame, coords): #to detect if the specific battle plates are present
        x1, y1, x2, y2 = coords
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        beige_mask = cv2.inRange(hsv, np.array([28, 20, 180]), np.array([30, 80, 255]))
        return np.count_nonzero(beige_mask) > 3500

    def get_name_via_ocr(self, plate_roi, is_opponent=True): #OCR implementation using psm 7 for linear line detection of text
        key = "OPP" if is_opponent else "MY"
        x1, y1, x2, y2 = self.name_offsets[key] # Corrdinates for covering/detecting names
        crop = plate_roi[y1:y2, x1:x2]
        if crop.size == 0: return "UNKNOWN"
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ' #hard-code to focus only on uppercase Alphabet, for easier detection of name3
        raw = pytesseract.image_to_string(thresh, config=config).upper().strip()
        
        clean = "".join(c for c in raw if c.isalpha())
        if len(clean) < 3: return "UNKNOWN"
        
        matches = difflib.get_close_matches(clean, self.pokemon_db.keys(), n=1, cutoff=0.51) #This will get the closes match of the OCR detection to a name in the jason database
        return matches[0] if matches else "UNKNOWN"

    def draw_hp_status(self, dashboard): #Display % health of opponent and my own pokemons
        #if self.current_state not in ["BATTLE", "SELECT_MOVE"]: return
        if self.current_state not in ["BATTLE", "SELECT_MOVE"]: return
        def gba_col(p): return (0, 255, 0) if p > 50 else (0, 255, 255) if p > 20 else (0, 0, 255) # Green if greater than 50, Yellow greater than 20, red to empty else
        
        for k, y, is_opp in [("OPP", 120, True), ("MY", 260, False)]:
            name = self.detected_names[k] # Gather name, hp and sprite of oppoenent and my own pokemon
            hp = int(self.hp_tracker[k])
            sprite = self.get_sprite(name, is_opp)
            
            cv2.putText(dashboard, f"{name} {hp}%", (20, y), 1, 1.4, (255, 255, 255), 2)
            if sprite is not None:
                self.overlay_sprite(dashboard, sprite, 280, y-70)
            
            #cv2.rectangle(dashboard, (20, y+50), (320, y+60), (40, 40, 40), -1)
            # Health level display. setting color to Green,Yellow, then Red based on current percentage tracked.
            cv2.rectangle(dashboard, (20, y+50), (20 + int(self.hp_tracker[k] * 3), y+60), gba_col(self.hp_tracker[k]), -1)

    def get_hp_percentage(self, frame, plate_coords, is_opponent=True): #detection of health bar pixels and updates according to current health in battle
        px1, py1, px2, py2 = plate_coords 
        plate_roi = frame[py1:py2, px1:px2]
        hx1, hy1, hx2, hy2 = (244, 118, 552, 122) if is_opponent else (228, 123, 534, 127)
        hp_bar_crop = plate_roi[hy1:hy2, hx1:hx2]
        if hp_bar_crop.size == 0: return None
        hsv = cv2.cvtColor(hp_bar_crop, cv2.COLOR_BGR2HSV)
        w, filled = hsv.shape[1], 0
        for x in range(w):
            if np.any((hsv[:, x][:, 1] > 70) & (hsv[:, x][:, 2] > 60)): filled += 1 # > 70 for color, and > 60 for brightness of pixel so not confused by dark shadows or black borders
            elif x > 5: break
        return (filled / w) * 100

    def highlight_hp_slots(self, frame, coords, is_opponent=True):
        px1, py1, px2, py2 = coords
        hx1, hy1, hx2, hy2 = (244, 118, 552, 122) if is_opponent else (228, 123, 534, 127)
        cv2.rectangle(frame, (px1 + hx1, py1 + hy1), (px1 + hx2, py1 + hy2), (0, 0, 255), 2) # Red rectangles over the hp of the pokemon

    def process_frame(self, frame): #all square frames positions on the screen and appear or not based on if in battle or not, and also specific frames
        self.frame_count += 1
        frame = cv2.resize(frame, (1920, 1080))
        display_game, dashboard = frame.copy(), np.zeros((1080, 400, 3), dtype=np.uint8)

        # Summary Detection
        s_y1, s_y2, s_x1, s_x2 = self.summary_roi
        hsv_s = cv2.cvtColor(frame[s_y1:s_y2, s_x1:s_x2], cv2.COLOR_BGR2HSV)
        is_summary = (np.count_nonzero(cv2.inRange(hsv_s, np.array([90, 150, 150]), np.array([110, 255, 220]))) > 12000) and \
                     (np.count_nonzero(cv2.inRange(hsv_s, np.array([0, 0, 245]), np.array([180, 10, 255]))) > 600)

        p_y1, p_y2, p_x1, p_x2 = self.party_roi
        #w_mask = cv2.inRange(cv2.cvtColor(frame[p_y1:p_y2, p_x1:p_x2], cv2.COLOR_BGR2HSV), np.array([0, 0, 255]), np.array([180, 5, 255]))
        py_mask = cv2.inRange(cv2.cvtColor(frame[p_y1:p_y2,p_x1:p_x2], cv2.COLOR_BGR2HSV), np.array([110, 50, 50]), np.array([120, 55, 255]))
        is_party = (np.count_nonzero(py_mask) > 300)#and (np.count_nonzero(bg_mask) > 2900)

        b_y1, b_y2, b_x1, b_x2 = self.bag_roi
        blue_mask = cv2.inRange(cv2.cvtColor(frame[b_y1:b_y2, b_x1:b_x2], cv2.COLOR_BGR2HSV), np.array([85, 220, 190]), np.array([99, 255, 200]))
        is_bag = (np.count_nonzero(blue_mask) > 350000) and not is_party

        # --- MOVE MENU DETECTION ---
        m_x1, m_y1, m_x2, m_y2 = self.move_menu_roi
        move_roi_img = frame[m_y1:m_y2, m_x1:m_x2]
        m_hsv = cv2.cvtColor(move_roi_img, cv2.COLOR_BGR2HSV)

        # Look for the white background
        move_bg_mask = cv2.inRange(m_hsv, np.array([0, 0, 240]), np.array([180, 15, 255]))
        
        # Check for if moves slot appears with that certain amount of white pixels
        is_slot_filled = np.count_nonzero(move_bg_mask) > 200000 # Threshold for move names to appear

        # Refined Move Menu Logic: White background present AND no command buttons AND move text exists
        is_move_menu = (np.count_nonzero(move_bg_mask) > 55000) and is_slot_filled 

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
            if is_move_menu:
                self.current_state = "SELECT_MOVE"
                """
                    Inside here to implement displaying
                    which move is effective or not
                """
                # --- NEW: Move Advantage Logic ---
                # 1. Crop the 'Type' box from the move menu
                tx1, ty1, tx2, ty2 = self.move_slots["Type"]
                type_roi = move_roi_img[ty1:ty2, tx1:tx2]
                
                # 2. OCR the type name
                gray_t = cv2.cvtColor(type_roi, cv2.COLOR_BGR2GRAY)
                _, thresh_t = cv2.threshold(gray_t, 150, 255, cv2.THRESH_BINARY_INV)
                type_raw = pytesseract.image_to_string(thresh_t, config='--psm 8').strip().upper() #To detect the type of the move that is currently hovered over (Single Word unlike psm 7)
                
                # 3. Detect opponent name and set default advantage to neutral for eventual calculation during match
                opp_name = self.detected_names["OPP"]
                advantage = 1.0
                if type_raw and opp_name != "UNKNOWN":
                    # Match the raw OCR string to valid types in your JSON
                    matches = difflib.get_close_matches(type_raw, self.type_chart.keys(), n=1, cutoff=0.55)
                    if matches:
                        advantage = self.moveAdvantage(matches[0], opp_name) # Calculation of effectiveness of a move against opponent typing

                # 4. Draw results on the Dashboard
                adv_text = f"EFFECTIVENESS: {advantage}x"
                adv_col = (0, 255, 0) if advantage > 1 else (0, 0, 255) if advantage < 1 else (255, 255, 255) #Can add in another color for 4x effective just need to get to a pokemon which has both typings a move is strong against.
                cv2.putText(dashboard, adv_text, (20, 450), 1, 1.5, adv_col, 2)
                cv2.rectangle(display_game, (m_x1, m_y1), (m_x2, m_y2), self.colors["SELECT_MOVE"], 3)
                # Draw the individual move squares
                for m_name, (ox1, oy1, ox2, oy2) in self.move_slots.items():
                    color = self.colors.get(m_name, (255, 255, 255))
                    cv2.rectangle(display_game, (m_x1+ox1, m_y1+oy1), (m_x1+ox2, m_y1+oy2), color, 2)
            else:
                self.current_state = "BATTLE" #If not going through move menue state will display Battle
            
            self.state_buffer = self.buffer_max
            for k, roi, is_opp in [("OPP", self.opp_roi, True), ("MY", self.my_roi, False)]:
                active = opp_active if is_opp else my_active
                if active:
                    """
                    If in battle we will check frame_count by the frame_rate (which in firered is 30 fps)
                    and it will be Unknown until it finds a detection of a name in the get_name_via_ocr method call
                    """
                    x1, y1, x2, y2 = roi
                    check_rate = 30 if self.detected_names[k] == "UNKNOWN" else 60
                    if self.frame_count % check_rate == 0:
                        res = self.get_name_via_ocr(frame[y1:y2, x1:x2], is_opp)
                        if res != "UNKNOWN": self.detected_names[k] = res

                    val = self.get_hp_percentage(frame, roi, is_opp)
                    if val is not None:
                        self.hp_tracker[k] = 0.0 if val < 1.5 else val

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
"""
cap captures the video being fed from OBS, or the directory containing the recorded mp4 video of the game to run
It will display in a separate window with a dashboard so you will have the emulator running the game, and a separate window of the game with the script executing
at the same time """
#cap = cv2.VideoCapture("/Users/jamesdracup/Desktop/Test.mp4")
cap = cv2.VideoCapture(0)
is_paused = False

while cap.isOpened():
    """Options to pause or resume the game"""
    if not is_paused:
        ret, frame = cap.read()
        if not ret: break
        out = bot.process_frame(frame)
        cv2.imshow("Pokemon Move Effectivness", cv2.resize(out, (1280, 600)))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): 
        is_paused = not is_paused
        print("PAUSED" if is_paused else "RESUMED")
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()