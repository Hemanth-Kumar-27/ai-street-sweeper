import threading
import customtkinter as ctk
from AI_Street_Sweeper.utils.video import VideoProcessor
from AI_Street_Sweeper.utils.image_converter import ImageConverter
from AI_Street_Sweeper.utils.predictor import DebrisPredictor
from AI_Street_Sweeper.utils.controller import Controller
from AI_Street_Sweeper.utils.power import PowerCalculator
from AI_Street_Sweeper.utils.segmentation import Segmentation

# =====================================================
# APPEARANCE
# =====================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# =====================================================
# COLORS
# =====================================================

BG_COLOR = "#141414"
MAIN_CARD = "#1F2A44"
HEADER = "#1B2954"
BORDER = "#2E7DFF"

PURPLE = "#A855F7"
BLUE = "#4DA3FF"
ORANGE = "#FF9F1C"
CYAN = "#33D9B2"
YELLOW = "#FFD43B"
GREEN = "#5BE37D"

TEXT = "#F5F5F5"

# =====================================================
# LOAD VIDEO
# =====================================================

video = VideoProcessor()

predictor = DebrisPredictor()

controller = Controller()

power = PowerCalculator()

segmenter = Segmentation()

# =====================================================
# VARIABLE
# =====================================================
prediction = "clean"
confidence = 0
frame_counter = 0
brush_rpm = 0
fan_rpm = 0
adaptive_power = 0
saving = 0
coverage = 0

inference_lock = threading.Lock()
model_results = {
    "prediction": prediction,
    "confidence": confidence,
    "coverage": coverage,
    "brush_rpm": brush_rpm,
    "fan_rpm": fan_rpm,
    "adaptive_power": adaptive_power,
    "saving": saving,
    "road_roi": None,
    "road_mask": None,
}

inference_thread = None

# =====================================================
# INFERENCE WORKER
# =====================================================

def run_inference(frame, roi, roi_mask):
    global model_results

    prediction_label, confidence_score = predictor.predict(
        roi
    )

    road_roi, road_mask, coverage_score = segmenter.process(
        frame,
        roi_mask
    )

    settings = controller.get_settings(
        prediction_label
    )

    brush_speed = settings["brush"]
    fan_speed = settings["fan"]

    power_result = power.calculate(
        brush_speed,
        fan_speed
    )

    with inference_lock:
        model_results.update({
            "prediction": prediction_label,
            "confidence": confidence_score,
            "coverage": coverage_score,
            "brush_rpm": brush_speed,
            "fan_rpm": fan_speed,
            "adaptive_power": power_result["power"],
            "saving": power_result["saving"],
            "road_roi": road_roi,
            "road_mask": road_mask,
        })


def start_inference(frame, roi, roi_mask):
    global inference_thread

    if inference_thread is None or not inference_thread.is_alive():
        inference_thread = threading.Thread(
            target=run_inference,
            args=(frame.copy(), roi.copy(), roi_mask.copy()),
            daemon=True
        )
        inference_thread.start()

# =====================================================
# WINDOW
# =====================================================

app = ctk.CTk()

app.title("AI Street Sweeper Control System")

app.state("zoomed")

app.configure(fg_color=BG_COLOR)


# =====================================================
# MAIN
# =====================================================

main = ctk.CTkFrame(
    app,
    fg_color=BG_COLOR
)

main.pack(
    fill="both",
    expand=True,
    padx=12,
    pady=12
)


# =====================================================
# HEADER
# =====================================================

header = ctk.CTkFrame(
    main,
    fg_color=HEADER,
    corner_radius=18,
    border_width=1,
    border_color=BORDER,
    height=85
)

header.pack(
    fill="x",
    pady=(5, 18)
)

header.pack_propagate(False)


title = ctk.CTkLabel(
    header,
    text="AI STREET SWEEPER CONTROL SYSTEM",
    font=("Segoe UI", 34, "bold"),
    text_color=TEXT
)

title.pack(expand=True)


# =====================================================
# VIDEO SECTION
# =====================================================

video_section = ctk.CTkFrame(
    main,
    fg_color=BG_COLOR
)

video_section.pack(
    fill="both",
    expand=True,
    pady=(0, 18)
)

video_section.grid_columnconfigure(0, weight=1)
video_section.grid_columnconfigure(1, weight=1)
video_section.grid_rowconfigure(0, weight=1)


# =====================================================
# ORIGINAL VIDEO
# =====================================================

left_panel = ctk.CTkFrame(
    video_section,
    fg_color=MAIN_CARD,
    corner_radius=18,
    border_width=1,
    border_color=BORDER
)

left_panel.grid(
    row=0,
    column=0,
    padx=(0, 10),
    sticky="nsew"
)


ctk.CTkLabel(
    left_panel,
    text="ORIGINAL VIDEO",
    text_color=PURPLE,
    font=("Segoe UI", 22, "bold")
).pack(pady=(15, 10))


video_display = ctk.CTkLabel(
    left_panel,
    text="ORIGINAL VIDEO",
    font=("Segoe UI", 20),
    text_color="#111111",
    fg_color="#D5DCE5",
    corner_radius=12
)

video_display.pack(
    fill="both",
    expand=True,
    padx=18,
    pady=(0, 18)
)


# =====================================================
# SEGMENTED ROAD / ROI
# =====================================================

right_panel = ctk.CTkFrame(
    video_section,
    fg_color=MAIN_CARD,
    corner_radius=18,
    border_width=1,
    border_color=BORDER
)

right_panel.grid(
    row=0,
    column=1,
    padx=(10, 0),
    sticky="nsew"
)


ctk.CTkLabel(
    right_panel,
    text="SEGMENTED ROAD (ROI)",
    text_color=CYAN,
    font=("Segoe UI", 22, "bold")
).pack(pady=(15, 10))


road_display = ctk.CTkLabel(
    right_panel,
    text="ROAD ROI",
    font=("Segoe UI", 20),
    text_color="#111111",
    fg_color="#D5DCE5",
    corner_radius=12
)

road_display.pack(
    fill="both",
    expand=True,
    padx=18,
    pady=(0, 18)
)


# =====================================================
# DASHBOARD
# =====================================================

dashboard = ctk.CTkFrame(
    main,
    fg_color=MAIN_CARD,
    corner_radius=14,
    border_width=1,
    border_color="#244C92",
    height=155
)

dashboard.pack(
    fill="x",
    padx=8,
    pady=(0, 8)
)

dashboard.pack_propagate(False)


# =====================================================
# CARD FUNCTION
# =====================================================

def create_card(title, color):

    frame = ctk.CTkFrame(
        dashboard,
        fg_color="#263653",
        corner_radius=14,
        border_width=1,
        border_color="#244C92"
    )

    frame.pack(
        side="left",
        fill="both",
        expand=True,
        padx=6,
        pady=8
    )

    ctk.CTkLabel(
        frame,
        text=title,
        font=("Segoe UI", 16, "bold"),
        text_color=color
    ).pack(pady=(12, 4))

    value = ctk.CTkLabel(
        frame,
        text="--",
        font=("Segoe UI", 30, "bold"),
        text_color=color
    )

    value.pack()

    bar = ctk.CTkProgressBar(
        frame,
        height=8,
        progress_color=color,
        fg_color="#555555"
    )

    bar.pack(
        fill="x",
        padx=18,
        pady=(8, 10)
    )

    bar.set(0)

    return value, bar


# =====================================================
# SIX PARAMETERS
# =====================================================

prediction_value, prediction_bar = create_card(
    "🟣 DEBRIS LEVEL",
    PURPLE
)

coverage_value, coverage_bar = create_card(
    "▧ COVERAGE",
    BLUE
)

brush_value, brush_bar = create_card(
    "🟠 BRUSH RPM",
    ORANGE
)

fan_value, fan_bar = create_card(
    "◉ FAN RPM",
    CYAN
)

power_value, power_bar = create_card(
    "⚡ POWER",
    YELLOW
)

saving_value, saving_bar = create_card(
    "🌿 ENERGY SAVING",
    GREEN
)


# =====================================================
# FOOTER
# =====================================================

footer = ctk.CTkFrame(
    main,
    fg_color=HEADER,
    height=28,
    corner_radius=8
)

footer.pack(
    fill="x",
    pady=(0, 2)
)

footer.pack_propagate(False)


ctk.CTkLabel(
    footer,
    text="● SYSTEM READY",
    text_color=GREEN,
    font=("Segoe UI", 12, "bold")
).pack(
    side="left",
    padx=12
)


ctk.CTkLabel(
    footer,
    text="AI Street Sweeper",
    text_color=CYAN,
    font=("Segoe UI", 12, "bold")
).pack(
    side="right",
    padx=12
)

# =====================================================
# VIDEO LOOP
# =====================================================

def update_video():

    global frame_counter
    global prediction
    global confidence
    global brush_rpm
    global fan_rpm
    global adaptive_power
    global saving
    global coverage

    frame = video.get_frame()

    if frame is None:
        app.after(30, update_video)
        return

    # -----------------------------
    # Original Video
    # -----------------------------

    image = ImageConverter.convert(frame)

    video_display.configure(
        image=image,
        text=""
    )

    video_display.image = image

    # -----------------------------
    # Trapezoid ROI
    # -----------------------------

    roi, roi_mask = video.get_roi(frame)

    # -----------------------------
    # CNN + SegFormer
    # -----------------------------

    frame_counter += 1

    if frame_counter % 20== 0:
        start_inference(
            frame,
            roi,
            roi_mask
        )

    with inference_lock:
        current_results = model_results.copy()

    if current_results["road_roi"] is not None:
        segmented_image = ImageConverter.convert(
            current_results["road_roi"]
        )

        road_display.configure(
            image=segmented_image,
            text=""
        )

        road_display.image = segmented_image

    prediction_value.configure(
        text=current_results["prediction"].upper()
    )

    prediction_bar.set({
        "clean": 0.25,
        "high": 1.00,
        "low": 0.50,
        "medium": 0.75
    }.get(
        current_results["prediction"].lower(),
        0
    ))

    coverage_value.configure(
        text=f"{current_results['coverage']:.1f}%"
    )

    coverage_bar.set(
        min(current_results["coverage"] / 100, 1.0)
    )

    brush_value.configure(
        text=f"{current_results['brush_rpm']} RPM"
    )

    brush_bar.set(
        min(current_results["brush_rpm"] / 340, 1.0)
    )

    fan_value.configure(
        text=f"{current_results['fan_rpm']} RPM"
    )

    fan_bar.set(
        min(current_results["fan_rpm"] / 2000, 1.0)
    )

    power_value.configure(
        text=f"{current_results['adaptive_power']:.2f} kW"
    )

    power_bar.set(
        min(current_results["adaptive_power"] / 3.2, 1.0)
    )

    saving_value.configure(
        text=f"{current_results['saving']:.1f} %"
    )

    saving_bar.set(
        min(current_results["saving"] / 100, 1.0)
    )

    app.after(
        10,
        update_video
    )

# =====================================================
# START
# =====================================================

update_video()

app.mainloop()