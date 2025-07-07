import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai
from google.genai import types
from PIL import Image
import io
from collections import Counter
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Hotel Room Amenities Analyzer",
    page_icon="🛏️",
    layout="wide"
)

# Application title and description
st.title("🛏️ Hotel Room Amenities Analyzer")
st.markdown("""
This application analyzes video or images from hotel rooms to identify missing or uncleaned amenities, helping to improve room quality and housekeeping.
""")

# Default API key - you can replace this with your actual key
DEFAULT_API_KEY = "AIzaSyDgEn3NhBExXJsYTU3R1S0NCwmV6Id7kqk"

# Function to extract frames from video
def extract_frames(video_path, num_frames=20):
    frames = []
    temp_image_paths = []
    
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        st.sidebar.info(f"Video duration: {duration:.2f} seconds")
        st.sidebar.info(f"Total frames: {total_frames}")
        
        # Calculate frame indices to extract (evenly distributed)
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Save frame temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_image_paths.append(temp_file.name)
                Image.fromarray(frame_rgb).save(temp_file.name)
                
        cap.release()
        return frames, temp_image_paths
    except Exception as e:
        st.error(f"Error extracting frames: {str(e)}")
        return [], []

# Function to analyze amenities in images using Gemini
# Now supports dynamic group selection per frame

def analyze_amenities(client, image_or_video, visible_groups=None):
    try:
        with open(image_or_video, 'rb') as f:
            img_bytes = f.read()
        # Build Gemini prompt dynamically based on visible_groups
        group_prompts = {
            "Bed & Pillows": "1. Bed & Pillows: Provide a single, detailed, actionable, and qualitative one-liner summary (15-25 words) describing the current state, issues, and what is correct or missing, in plain English.",
            "Toilet & Toiletries/Towel": "2. Toilet & Toiletries/Towel: Provide a single, detailed, actionable, and qualitative one-liner summary (15-25 words) describing the current state, issues, and what is correct or missing, in plain English.",
            "Mirror": "3. Mirror: Provide a single, detailed, actionable, and qualitative one-liner summary (15-25 words) describing the current state, issues, and what is correct or missing, in plain English.",
            "Room Clutter": "4. Room Clutter: Provide a single, detailed, actionable, and qualitative one-liner summary (15-25 words) describing the current state, issues, and what is correct or missing, in plain English."
        }
        if visible_groups is None:
            visible_groups = list(group_prompts.keys())
        prompt_sections = [group_prompts[g] for g in visible_groups if g in group_prompts]
        gemini_prompt = (
            "You are an expert hotel inspector. Given the following room image or video, analyze the state of the room and amenities.\n" +
            "\n".join(prompt_sections) +
            "\nOnly mention items visible in the image/video. Do not mention or speculate about groups that are not visible."
        )
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type='image/jpeg',
                ),
                gemini_prompt
            ]
        )
        print(f"Raw response for {os.path.basename(image_or_video)}:")
        print(response.text)
        # Return the plain English summary for display
        return {"summary": response.text.strip()}
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return {"summary": "Could not analyze image."}

# Function to visualize room amenities analysis
# (No longer used for summary display, kept for legacy support)
def visualize_amenities_data(all_amenities_data):
    pass  # Deprecated: use show_overall_housekeeping_summary instead

# Function to build and display the overall housekeeping summary box

def show_overall_housekeeping_summary(food_data):
    all_summaries = [data.get("summary", "") for data in food_data if data.get("summary")]
    present_items = set()
    missing_items = set()
    # For action required extraction
    action_required = []
    # For inspection checklist extraction
    checklist = {
        "Bed Made": None,
        "Pillows Arranged": None,
        "No Clutter": None,
        "Extra Pillow/Blanket Present": None,
        "Toilet Cleanliness": None,
        "Toilet Flush Lid Position": None,
        "Mirror Present": None,
        "Mirror Cleanliness": None,
        "Toiletries and Towel Present": None
    }
    # Track if bathroom or mirror is not shown
    not_shown = {"Toiletries and Towel Present": False, "Mirror Present": False, "Toilet": False}
    # Parse all summaries for actionable items and checklist
    for summary in all_summaries:
        lines = summary.splitlines()
        in_present_section = False
        in_missing_section = False
        for idx, line in enumerate(lines):
            l = line.strip().lower()
            if l.startswith("1. room amenities present"):
                in_present_section = True
                in_missing_section = False
                continue
            if l.startswith("2. missing"):
                in_present_section = False
                in_missing_section = True
                continue
            if (line.strip() and line.strip()[0].isdigit() and line.strip()[1] == "." and not l.startswith("1. room amenities present") and not l.startswith("2. missing")):
                in_present_section = False
                in_missing_section = False
                continue
            if in_present_section and line.strip() and not l.startswith("room amenities present"):
                present_items.add(line.strip())
                # Explicitly mark present/correct items as True
                for item in checklist:
                    if item.lower().replace("/", " ") in line.lower():
                        checklist[item] = True
            if in_missing_section and line.strip() and not l.startswith("missing") and not l.startswith("items"):
                missing_items.add(line.strip())
                lcase = line.lower()
                # Try to map to checklist/action required
                if "wrinkle" in lcase or "not pulled taut" in lcase:
                    checklist["Bed Made"] = False
                    action_required.append("Smooth out wrinkles on the bed covers and ensure they are pulled taut.")
                if "pillow" in lcase and ("not plump" in lcase or "not arranged" in lcase or "out of place" in lcase):
                    checklist["Pillows Arranged"] = False
                    action_required.append("Plump and neatly arrange all pillows on the bed.")
                if ("extra pillow" in lcase or "blanket" in lcase) and ("not present" in lcase or "not available" in lcase or "missing" in lcase):
                    checklist["Extra Pillow/Blanket Present"] = False
                    action_required.append("Ensure an extra pillow and blanket are placed in the wardrobe or visibly available.")
                if ("toilet" in lcase and ("dirty" in lcase or "not clean" in lcase or "unclean" in lcase)):
                    checklist["Toilet Cleanliness"] = False
                    action_required.append("Clean the toilet thoroughly.")
                if ("toilet flush lid" in lcase and ("open" in lcase or "not closed" in lcase)):
                    checklist["Toilet Flush Lid Position"] = False
                    action_required.append("Ensure the toilet flush lid is closed.")
                if ("mirror" in lcase and ("not present" in lcase or "missing" in lcase)):
                    checklist["Mirror Present"] = False
                    action_required.append("Ensure a mirror is present in the bathroom.")
                if ("mirror" in lcase and ("dirty" in lcase or "not clean" in lcase or "unclean" in lcase)):
                    checklist["Mirror Cleanliness"] = False
                    action_required.append("Clean the mirror to ensure it is spotless.")
                if ("bathroom" in lcase and ("no toiletries" in lcase or "no towel" in lcase or "no soap" in lcase or "no equipment" in lcase or "missing equipment" in lcase)):
                    checklist["Toiletries and Towel Present"] = False
                    action_required.append("Ensure all standard toiletries and towels are present.")
            # Check for not shown bathroom or mirror
            if in_missing_section and line.strip() and not l.startswith("missing") and not l.startswith("items"):
                lcase = line.lower()
                if ("bathroom" in lcase and ("not visible" in lcase or "not shown" in lcase)):
                    not_shown["Toiletries and Towel Present"] = True
                if ("mirror" in lcase and ("not visible" in lcase or "not shown" in lcase)):
                    not_shown["Mirror Present"] = True
    # --- Extract qualitative commentary for key amenities from Gemini output ---
    # We'll look for lines in the summary that mention these amenities and use them for the checklist one-liners
    commentary_targets = {
        "Pillows Arranged": ["pillow", "pillows"],
        "Toilet Cleanliness": ["toilet"],
        "Toilet Flush Lid Position": ["flush lid", "toilet lid", "flush cover"],
        "Toiletries and Towel Present": ["toiletries", "towel", "soap"],
        "Bed Made": ["bed made", "bed covers", "bed sheet", "bed is made"],
        "No Clutter": ["clutter", "bags", "clothing"],
        "Extra Pillow/Blanket Present": ["extra pillow", "blanket"],
        "Mirror Cleanliness": ["mirror"],
        "Bathroom Floor Dryness": ["bathroom floor", "floor dryness", "floor is dry", "wet floor"]
    }
    # --- Detect if toilet is not present at all ---
    toilet_not_present = False
    bathroom_not_present = False
    mirror_not_present = False
    for summary in all_summaries:
        lines = summary.splitlines()
        for line in lines:
            l = line.strip().lower()
            if any(x in l for x in ["no toilet", "toilet not visible", "toilet not present", "no bathtub, toilet"]):
                toilet_not_present = True
            if any(x in l for x in ["bathroom not visible", "bathroom not shown", "cannot assess the cleanliness or status of bathtub", "cannot assess the cleanliness or status of bathroom"]):
                bathroom_not_present = True
            if any(x in l for x in ["mirror not visible", "mirror not shown", "cannot assess the cleanliness or status of mirror"]):
                mirror_not_present = True

    # --- If key areas are not present, do NOT show dependent checklist or actions ---
    checklist_items_to_show = list(checklist.keys())
    if toilet_not_present or bathroom_not_present:
        for dep in ["Toilet Cleanliness", "Toilet Flush Lid Position", "Toiletries and Towel Present"]:
            if dep in checklist_items_to_show:
                checklist_items_to_show.remove(dep)
        action_required.append("Toilet/bathroom area not visible or not assessable. Please capture this area in future scans.")
    if mirror_not_present:
        for dep in ["Mirror Present", "Mirror Cleanliness"]:
            if dep in checklist_items_to_show:
                checklist_items_to_show.remove(dep)
        action_required.append("Mirror area not visible or not assessable. Please capture this area in future scans.")

    # --- If most amenities are not visible, show a single summary line ---
    # Fix: Only suppress summary if no checklist items are available
    if not checklist_items_to_show:
        st.markdown(f"""
        <div style='border:2px solid #4F8BF9; border-radius:14px; padding:24px 18px 18px 18px; background:linear-gradient(135deg,#1e2a3a 80%,#2d3e5e 100%); margin-top:20px; color:#fff; box-shadow: 0 4px 24px #0002;'>
            <div style='font-size: 22px; font-weight: bold; margin-bottom: 18px; letter-spacing: 0.5px;'>Overall Housekeeping Summary</div>
            <div style='font-size: 16px; color: #e3eaf5;'>
                Unable to assess most amenities due to limited or incomplete room view. Please ensure all key areas (bathroom, toilet, mirror, bed, etc.) are visible in the images or video for a complete inspection.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    # --- Commentary extraction (only for visible amenities) ---
    checklist_summaries = {
        "Bed Made": "The bed covers are smooth and pulled taut.",
        "Pillows Arranged": "Pillow count and placement: All pillows are present, plumped, and neatly arranged.",
        "Extra Pillow/Blanket Present": "An extra pillow and blanket are visibly available.",
        "Toilet Cleanliness": "Bathroom cleanliness: Toilet and bathroom appear clean and well-maintained.",
        "Toilet Flush Lid Position": "Toilet flush lid position: The lid is closed as required.",
        "Mirror Present": "Mirror is present or not visible in the images.",
        "Mirror Cleanliness": "Mirror cleanliness: The mirror appears spotless and streak-free.",
        "Toiletries and Towel Present": "Toiletries and towels: All standard toiletries and towels are present and neatly arranged."
    }
    for summary in all_summaries:
        lines = summary.splitlines()
        for item, keywords in commentary_targets.items():
            if toilet_not_present and item in ["Toilet Cleanliness", "Toilet Flush Lid Position", "Toiletries and Towel Present"]:
                continue  # Skip commentary for these if toilet not present
            for line in lines:
                l = line.strip().lower()
                if any(kw in l for kw in keywords):
                    # Use the first matching line as the commentary
                    checklist_summaries[item] = line.strip()
                    break
    # Fill in present items for checklist with one-liner summary
    checklist_items_to_show = list(checklist.keys())
    for item in checklist:
        if not_shown.get(item, False):
            checklist_summaries[item] = f"{item.split()[0]} not shown in the images."
            checklist[item] = None  # Will trigger ! mark
            action_required.append(f"{item.split()[0]} not shown in the images. Please ensure to capture it in future scans.")
            # If bathroom not shown, remove toiletries/towel from checklist
            if item == "Toiletries and Towel Present":
                if "Toiletries and Towel Present" in checklist_items_to_show:
                    checklist_items_to_show.remove("Toiletries and Towel Present")
            # If mirror not shown, remove mirror cleanliness from checklist
            if item == "Mirror Present":
                if "Mirror Cleanliness" in checklist_items_to_show:
                    checklist_items_to_show.remove("Mirror Cleanliness")
        elif checklist[item] is None:
            # If not flagged as missing, provide a positive, qualitative summary
            if item == "Bed Made":
                checklist_summaries[item] = "The bed covers are smooth and pulled taut."
            elif item == "Pillows Arranged":
                checklist_summaries[item] = "Pillow count and placement: All pillows are present, plumped, and neatly arranged."
            elif item == "Extra Pillow/Blanket Present":
                checklist_summaries[item] = "An extra pillow and blanket are visibly available."
            elif item == "Toilet Cleanliness":
                if not_shown.get("Toilet Cleanliness", False):
                    checklist_summaries[item] = "Toilet not shown in the images."
                    checklist[item] = None
                else:
                    checklist_summaries[item] = "Bathroom cleanliness: Toilet and bathroom appear clean and well-maintained."
            elif item == "Toilet Flush Lid Position":
                checklist_summaries[item] = "Toilet flush lid position: The lid is closed as required."
            elif item == "Mirror Present":
                checklist_summaries[item] = "Mirror is present or not visible in the images."
            elif item == "Mirror Cleanliness":
                checklist_summaries[item] = "Mirror cleanliness: The mirror appears spotless and streak-free."
            elif item == "Toiletries and Towel Present":
                checklist_summaries[item] = "Toiletries and towels: All standard toiletries and towels are present and neatly arranged."
    # --- Remove dependent items if bathroom/toilet is not shown ---
    # If bathroom/toilet is not shown, remove all related checklist items and actions
    bathroom_not_shown = not_shown.get("Toiletries and Towel Present", False) or not_shown.get("Toilet Cleanliness", False)
    if bathroom_not_shown:
        # Remove dependent checklist items
        for dep in ["Toilet Cleanliness", "Toilet Flush Lid Position", "Toiletries and Towel Present"]:
            if dep in checklist_items_to_show:
                checklist_items_to_show.remove(dep)
        # Remove related actions
        action_required = [a for a in action_required if not any(x in a.lower() for x in ["toilet", "flush lid", "toiletries", "bathroom"]) ]
        # Add a one-liner to action required
        action_required.append("Bathroom/toilet area not shown in images. Please capture this area in future scans.")
        # Remove commentary for these items
        for dep in ["Toilet Cleanliness", "Toilet Flush Lid Position", "Toiletries and Towel Present"]:
            checklist_summaries[dep] = "Not shown in images."

    # --- Remove mirror cleanliness if mirror not shown ---
    if not_shown.get("Mirror Present", False):
        if "Mirror Cleanliness" in checklist_items_to_show:
            checklist_items_to_show.remove("Mirror Cleanliness")
        action_required = [a for a in action_required if "mirror" not in a.lower()]
        checklist_summaries["Mirror Cleanliness"] = "Not shown in images."

    # --- Improved qualitative commentary extraction and consistency for missing amenities ---
    # Parse Gemini output for each amenity, capturing both present and missing commentary
    checklist_items_to_show = []
    checklist_summaries = {}
    action_required = []
    not_shown = {"Toiletries and Towel Present": False, "Mirror Present": False, "Toilet": False}

    # Build a mapping of amenity keywords to checklist keys
    amenity_keywords = {
        "Bed Made": ["bed made", "bed covers", "bed sheet", "bed is made", "unmade bed", "bed appears unmade"],
        "Pillows Arranged": ["pillow", "pillows"],
        "Extra Pillow/Blanket Present": ["extra pillow", "blanket"],
        "Toilet Cleanliness": ["toilet"],
        "Toilet Flush Lid Position": ["flush lid", "toilet lid", "flush cover"],
        "Toiletries and Towel Present": ["toiletries", "towel", "soap"],
        "Mirror Present": ["mirror present", "mirror is present", "mirror not present", "mirror missing"],
        "Mirror Cleanliness": ["mirror clean", "mirror cleanliness", "mirror dirty", "mirror not present", "mirror missing"],
    }

    # Collect all lines from all summaries for easier searching
    all_lines = []
    for summary in all_summaries:
        all_lines.extend([line.strip() for line in summary.splitlines() if line.strip()])

    # For each amenity, find the most relevant qualitative line
    missing_amenities = set()
    for item, keywords in amenity_keywords.items():
        found = False
        for line in all_lines:
            l = line.strip().lower()
            if any(kw in l for kw in keywords):
                checklist_summaries[item] = line
                found = True
                # If negative/absent, add to action required
                if item == "No Clutter":
                    # Only mark as missing if explicit negative about clutter
                    if any(x in l for x in ["clutter", "bags", "clothing", "personal items", "visible clutter", "scattered around", "mess", "untidy"]):
                        if any(x in l for x in ["no clutter", "tidy", "free of clutter", "no bags", "no clothing", "no personal items", "room is tidy"]):
                            checklist[item] = True
                        else:
                            missing_amenities.add(item)
                            checklist[item] = False
                            action_required.append("Remove any visible clutter or misplaced items from the room.")
                    else:
                        checklist[item] = True
                else:
                    # Improved: For Bed & Pillows group, only mark as missing if any of its items are explicitly missing/incorrect
                    if item in ["Bed Made", "Pillows Arranged", "Extra Pillow/Blanket Present"]:
                        if any(x in l for x in ["not present", "missing", "not shown", "not visible", "no "+keywords[0].split()[0], "not clean", "dirty", "not plump", "not arranged", "wrinkle", "not pulled taut", "out of place", "open"]):
                            missing_amenities.add(item)
                            checklist[item] = False
                            if item in amenity_fallbacks:
                                action_required.append(amenity_fallbacks[item][1])
                            else:
                                action_required.append(f"Address issue with: {item}.")
                        else:
                            checklist[item] = True  # Mark as present/correct
                    else:
                        if any(x in l for x in ["not present", "missing", "not shown", "not visible", "no "+keywords[0].split()[0], "not clean", "dirty", "not plump", "not arranged", "wrinkle", "not pulled taut", "out of place", "open"]):
                            missing_amenities.add(item)
                            checklist[item] = False
                            if item in amenity_fallbacks:
                                action_required.append(amenity_fallbacks[item][1])
                            else:
                                action_required.append(f"Address issue with: {item}.")
                        else:
                            checklist[item] = True  # Mark as present/correct
                break
        if not found:
            if item == "No Clutter":
                checklist_summaries[item] = "Room is tidy and free of visible clutter, with no bags, clothing, or personal items left behind."
                checklist[item] = True
            elif item in amenity_fallbacks:
                checklist_summaries[item] = amenity_fallbacks[item][0]
                missing_amenities.add(item)
                checklist[item] = False
                action_required.append(amenity_fallbacks[item][1])
        if item in checklist_summaries:
            checklist_items_to_show.append(item)
    # --- Remove dependent items if toilet/bathroom is not visible
    toilet_unseen = any(a in missing_amenities for a in ["Toilet Cleanliness", "Toilet Flush Lid Position", "Toiletries and Towel Present"])
    mirror_unseen = any(a in missing_amenities for a in ["Mirror Present", "Mirror Cleanliness"])

    if toilet_unseen:
        for dep in ["Toilet Cleanliness", "Toilet Flush Lid Position", "Toiletries and Towel Present"]:
            if dep in checklist_items_to_show:
                checklist_items_to_show.remove(dep)
        action_required.append("Toilet/bathroom area not visible. Please capture this area in future scans.")
    if mirror_unseen:
        for dep in ["Mirror Present", "Mirror Cleanliness"]:
            if dep in checklist_items_to_show:
                checklist_items_to_show.remove(dep)
        action_required.append("Mirror area not visible. Please capture this area in future scans.")

    # --- Grouped checklist sections ---
    grouped_checklist = [
        {
            "title": "Bed & Pillows",
            "items": ["Bed Made", "Pillows Arranged", "Extra Pillow/Blanket Present"],
            "icon": "🛏️"
        },
        {
            "title": "Toilet & Toiletries/Towel",
            "items": ["Toilet Cleanliness", "Toilet Flush Lid Position", "Toiletries and Towel Present"],
            "icon": "🚽"
        },
        {
            "title": "Mirror",
            "items": ["Mirror Present", "Mirror Cleanliness"],
            "icon": "🪞"
        },
        {
            "title": "Room Clutter",
            "items": ["No Clutter"],
            "icon": "🧹"
        }
    ]

    # --- Determine which groups are visible in at least one frame ---
    # A group is visible if any of its items are present in checklist_summaries
    visible_groups = set()
    for group in grouped_checklist:
        for item in group["items"]:
            if item in checklist_summaries:
                visible_groups.add(group["title"])
                break
    # --- Ensure Room Clutter group is always visible (green if no clutter info, pink if clutter detected) ---
    if "Room Clutter" not in visible_groups:
        # If No Clutter is not in checklist_summaries, add a default positive summary and mark as present
        if "No Clutter" not in checklist_summaries:
            checklist_summaries["No Clutter"] = "Room is tidy and free of visible clutter, with no bags, clothing, or personal items left behind."
            checklist["No Clutter"] = True
        visible_groups.add("Room Clutter")

    # --- Extract qualitative commentary for each group (improved, mutually exclusive, Gemini-driven) ---
    group_commentary = {}
    group_missing = {}
    for group in grouped_checklist:
        commentary_lines = []
        # For Bed & Pillows: mark green if there is no issue with any of its items (all True or None)
        if group["title"] == "Bed & Pillows":
            # Mark as missing (pink) only if any item is explicitly False
            missing = any(checklist.get(item) is False for item in group["items"])
            # If all are True or None (i.e., no explicit False), always green
            if not missing:
                missing = False
        # For Room Clutter: only pink if clutter is detected (i.e., checklist["No Clutter"] is False)
        elif group["title"] == "Room Clutter":
            missing = checklist.get("No Clutter") is False
        else:
            missing = any(checklist.get(item) is False for item in group["items"])
        for item in group["items"]:
            comm = checklist_summaries.get(item)
            if comm:
                # For Toilet & Toiletries/Towel, filter out lines about bed, mirror, or clutter
                if group["title"] == "Toilet & Toiletries/Towel":
                    if any(x in comm.lower() for x in ["bed", "pillow", "mirror", "clutter", "headboard"]):
                        continue
                # For Bed & Pillows, filter out lines about toilet, toiletries, mirror, clutter
                if group["title"] == "Bed & Pillows":
                    if any(x in comm.lower() for x in ["toilet", "bathroom", "towel", "toiletries", "mirror", "clutter"]):
                        continue
                # For Mirror, filter out lines about bed, toilet, toiletries, clutter
                if group["title"] == "Mirror":
                    if any(x in comm.lower() for x in ["bed", "pillow", "toilet", "bathroom", "towel", "toiletries", "clutter"]):
                        continue
                # For Room Clutter, filter out lines about bed, toilet, toiletries, mirror
                if group["title"] == "Room Clutter":
                    if any(x in comm.lower() for x in ["bed", "pillow", "toilet", "bathroom", "towel", "toiletries", "mirror"]):
                        continue
                if comm not in commentary_lines:
                    commentary_lines.append(comm)
        # Synthesize a single line for the group
        if commentary_lines:
            group_commentary[group["title"]] = commentary_lines[0]
        else:
            # If none of the group's items are visible, show a not visible message
            all_none = all(checklist.get(item) is None for item in group["items"])
            if all_none:
                group_commentary[group["title"]] = f"{group['title']} is not visible in this image, so its condition cannot be assessed."
                missing = True
            else:
                # Fallbacks for each group (only if Gemini output is not available)
                if group["title"] == "Bed & Pillows":
                    if not missing:
                        group_commentary[group["title"]] = "The bed is neatly made with smooth covers, all pillows are plumped and arranged, and an extra pillow/blanket is available as required."
                    else:
                        group_commentary[group["title"]] = "Bed covers are wrinkled or not pulled taut, pillows are not plumped or arranged, or extra pillow/blanket is missing. Smooth covers, arrange pillows, and ensure extra bedding is available."
                elif group["title"] == "Toilet & Toiletries/Towel":
                    if not missing:
                        group_commentary[group["title"]] = "Toilet and bathroom are clean, flush lid is closed, and all standard toiletries and towels are present and neatly arranged."
                    else:
                        group_commentary[group["title"]] = "Toilet or bathroom is not clean, flush lid is open, or toiletries/towels are missing. Clean thoroughly, close lid, and restock amenities."
                elif group["title"] == "Mirror":
                    if not missing:
                        group_commentary[group["title"]] = "Mirror is present and spotless, with no visible streaks or marks, providing a clear reflection for guests."
                    else:
                        group_commentary[group["title"]] = "Mirror is missing or dirty. Ensure a mirror is present and clean it thoroughly for a streak-free finish."
                elif group["title"] == "Room Clutter":
                    if not missing:
                        group_commentary[group["title"]] = "Room is tidy and free of visible clutter, with no bags, clothing, or personal items left behind."
                    else:
                        group_commentary[group["title"]] = "Room has visible clutter such as bags, clothing, or personal items. Remove all clutter and tidy the room for the next guest."
        group_missing[group["title"]] = missing

    # --- Build grouped checklist HTML ---
    checklist_html = "<div style='background: #25344b; border-radius: 14px; padding: 18px 22px; margin-bottom: 18px; box-shadow: 0 2px 8px #0002;'>"
    checklist_html += "<div style='color: #7ecfff; font-weight: 600; font-size: 17px; margin-bottom: 10px;'><span style='margin-right:8px;'>🗹</span>Inspection Checklist</div>"
    checklist_html += "<div style='display: flex; flex-direction: column; gap: 12px; margin-top: 8px;'>"
    for group in grouped_checklist:
        title = group["title"]
        if title not in visible_groups:
            continue  # Only show visible groups
        icon = group["icon"]
        commentary = group_commentary[title]
        missing = group_missing[title]
        color = "#ffcccc" if missing else "#e6ffe6"
        checklist_html += f"<div style='background: {color}; border-radius: 8px; padding: 10px 14px; color: #222; display: flex; align-items: flex-start; gap: 10px;'>"
        checklist_html += f"<span style='font-size: 20px; margin-top: 2px;'>{icon}</span>"
        checklist_html += f"<div><b>{title}</b><br><span style='font-size: 15px;'>{commentary}</span></div>"
        checklist_html += "</div>"
    checklist_html += "</div></div>"

    # --- Build Action Required section (one per missing group) ---
    action_required = []
    for group in grouped_checklist:
        title = group["title"]
        if title not in visible_groups:
            continue
        if group_missing[title]:
            if title == "Bed & Pillows":
                action_required.append("Smooth out wrinkles, make the bed, and arrange all pillows neatly. Ensure extra pillow/blanket is available.")
            elif title == "Toilet & Toiletries/Towel":
                action_required.append("Clean the toilet, close the flush lid, and ensure all toiletries and towels are present and arranged.")
            elif title == "Mirror":
                action_required.append("Ensure a clean, visible mirror is present in the room.")
            elif title == "Room Clutter":
                action_required.append("Remove any visible clutter or misplaced items from the room.")
    if not action_required:
        action_required.append("No immediate action required.")

    # --- Build Action Required HTML ---
    action_html = "<div style='background: #25344b; border-radius: 14px; padding: 18px 22px; margin-bottom: 18px; box-shadow: 0 2px 8px #0002;'>"
    action_html += "<div style='color: #7ecfff; font-weight: 600; font-size: 17px; margin-bottom: 10px;'><span style='margin-right:8px;'>🛈</span>Action Required</div>"
    action_html += "<ol style='color: #e3eaf5; font-size: 15px; margin-left: 18px; padding-left: 0;'>"
    for item in action_required:
        action_html += f"<li>{item}</li>"
    action_html += "</ol></div>"

    # --- Final output box styled as in the image ---
    st.markdown(f"""
    <div style='display: flex; gap: 32px; margin-top: 18px;'>
        <div style='flex: 1 1 0;'>
            <div style='font-size: 22px; font-weight: bold; margin-bottom: 18px; letter-spacing: 0.5px;'>Overall Housekeeping Summary</div>
            {action_html}
        </div>
        <div style='flex: 1 1 0;'>
            {checklist_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    return
    # --- Remove legacy checklist/action logic below (now replaced by grouped_checklist logic) ---
    # ...existing code...
    # --- Amenity-specific fallback one-liners and actions
amenity_fallbacks = {
    "Bed Made": ("The bed covers are wrinkled and not pulled taut.", "Smooth out wrinkles on the bed covers and ensure they are pulled taut."),
    "Pillows Arranged": ("Pillows are not plumped or neatly arranged; one is visibly out of place.", "Plump and neatly arrange all pillows on the bed."),
    "Extra Pillow/Blanket Present": ("No extra pillow or blanket is visibly available in the room.", "Ensure an extra pillow and blanket are placed in the wardrobe or visibly available."),
    "Toilet Cleanliness": ("Toilet appears dirty or unclean, or cleanliness could not be determined from the images.", "Clean the toilet and ensure it is spotless."),
    "Toilet Flush Lid Position": ("The toilet flush lid is open or not closed.", "Close the toilet flush lid as required."),
    "Toiletries and Towel Present": ("Some or all standard toiletries and towels are missing or not arranged.", "Arrange all standard toiletries and towels neatly in the bathroom."),
    "Mirror Present": ("Mirror not present or not visible in the images.", "Ensure a mirror is present and visible in the room."),
    "Mirror Cleanliness": ("The mirror appears dirty, streaked, or not clean.", "Clean the mirror until it is spotless and streak-free.")
}

def main():
    # Setup sidebar
    st.sidebar.header("Settings")
    
    # Enter API key (pre-filled with default)
    api_key = st.sidebar.text_input("Gemini API Key", value=DEFAULT_API_KEY, type="password")
    
    # Initialize the client
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            st.sidebar.success("API Key accepted")
        except Exception as e:
            st.sidebar.error(f"Error with API key: {str(e)}")
            return
    else:
        st.sidebar.warning("Please enter your Gemini API Key")
        return
    
    # Options for analyzing video or image
    analysis_option = st.sidebar.radio(
        "Select Input Type:",
        ("Upload Video", "Upload Images")
    )
    
    # Initialize session state for tracking analysis
    if 'analyzed_frames' not in st.session_state:
        st.session_state.analyzed_frames = []
    if 'food_data' not in st.session_state:
        st.session_state.food_data = []
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    
    # Clear previous analysis button
    if st.sidebar.button("Clear Previous Analysis"):
        # Clean up temporary files
        for temp_file in st.session_state.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Reset session state
        st.session_state.analyzed_frames = []
        st.session_state.food_data = []
        st.session_state.temp_files = []
        st.rerun()
    if analysis_option == "Upload Video":
        # Video upload
        uploaded_file = st.file_uploader("Upload a room video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            st.session_state.temp_files.append(temp_file.name)
            with st.spinner("Extracting frames from video..."):
                frames, temp_image_paths = extract_frames(temp_file.name)
                st.session_state.temp_files.extend(temp_image_paths)
            if frames:
                st.success(f"Successfully extracted {len(frames)} frames from video!")
                st.subheader("Extracted Room Frames")
                cols = st.columns(5)
                for i, frame in enumerate(frames):
                    cols[i % 5].image(frame, caption=f"Frame {i+1}", use_container_width=True)
                if st.button("Analyze Room Amenities in Frames"):
                    progress_bar = st.progress(0)
                    st.session_state.food_data = []
                    for i, image_path in enumerate(temp_image_paths):
                        progress_text = st.empty()
                        progress_text.text(f"Analyzing frame {i+1}/{len(temp_image_paths)}...")
                        food_data = analyze_amenities(client, image_path)
                        st.session_state.food_data.append(food_data)
                        progress_bar.progress((i + 1) / len(temp_image_paths))
                    progress_text.text("Analysis complete!")
                    progress_bar.progress(100)
                if st.session_state.food_data:
                    st.header("Room Amenities Analysis Results")
                    show_overall_housekeeping_summary(st.session_state.food_data)
    else:
        uploaded_images = st.file_uploader("Upload room images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if uploaded_images:
            st.subheader("Uploaded Room Images")
            cols = st.columns(5)
            temp_image_paths = []
            for i, img_file in enumerate(uploaded_images):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(img_file.getvalue())
                temp_file.close()
                temp_image_paths.append(temp_file.name)
                st.session_state.temp_files.append(temp_file.name)
                cols[i % 5].image(img_file, caption=f"Image {i+1}", use_container_width=True)
            if st.button("Analyze Room Amenities in Images"):
                progress_bar = st.progress(0)
                st.session_state.food_data = []
                for i, image_path in enumerate(temp_image_paths):
                    progress_text = st.empty()
                    progress_text.text(f"Analyzing image {i+1}/{len(temp_image_paths)}...")
                    food_data = analyze_amenities(client, image_path)
                    st.session_state.food_data.append(food_data)
                    progress_bar.progress((i + 1) / len(temp_image_paths))
                progress_text.text("Analysis complete!")
                progress_bar.progress(100)
                if st.session_state.food_data:
                    st.header("Room Amenities Analysis Results")
                    show_overall_housekeeping_summary(st.session_state.food_data)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This application helps hotels identify and improve room quality by analyzing visual data for missing or uncleaned amenities.\nUpload videos or images of hotel rooms to get actionable insights.
        """
    )

if __name__ == "__main__":
    main()