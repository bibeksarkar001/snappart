"""Frozen settings of the capture pipeline, chosen on held-out data (v1 stages 1, 2, 5) and the v2 feature stages. Change only with a new measurement."""

# frames (v1 stage 1)
SCAN_STRIDE = 2                 # marker detection on every 2nd frame
MIN_MARKERS = 4                 # a pose needs >= 4 markers
MAX_RMS_PX = 1.0                # reject frames with marker reprojection rms above this ...
MAX_RMS_RELAXED_PX = 1.5        # ... relaxed in 0.25 px steps up to this only if fewer than MIN_ACCEPTED frames pass (video 3 needed 1.25)
MIN_ACCEPTED = 160
FRAMES_PER_VIDEO = 80           # selected views per video, spread over azimuth x elevation
CALIB_FRAMES_PER_VIDEO = 120    # frames per video for the joint mat self-calibration
MAT_PRIOR_PX_PER_MM = 2.0       # bundle-adjustment prior pulling marker corners toward the printed layout

# masks (v1 stage 2)
SAM_CONCEPTS = ("object", "plastic part", "part")   # generic: main concept, then fallbacks for frames it misses (a job's part name goes first)
PART_HUE_DEFAULT = (38.0, 80.0)  # Lab hue window (deg) of the part when no colour model is learned yet (masks.learn_colour learns it per job)
MIN_COLOUR_IN_SAM = 0.85
MIN_SAM_CONF = 0.5

# registration (v3 stage 3): Powell refinement voxel. 1.0 mm (24 s) stays within 0.125 mm of the 0.5 mm-grid result (128 s) on every video;
# against the v1 cached transform it is 0.29 mm (0.5 mm grid: 0.19). A coarse-to-fine shell refinement was tried: 113 s, no gain.
REG_REFINE_VOXEL_MM = 1.0

# carve (v1 stage 5, held-out IoU)
K_ROBUST = 12                   # a voxel is outside only if 12 views put it outside ...
K_TUNED_AT_VIEWS = 229          # ... tuned with 229 views; pipeline.k_for() scales it with the view count
GROW_MM = 0.1                   # every silhouette dilated by 0.1 mm
VOXEL_MM = 0.15
TAUBIN_ITERS = 5
BOX_LO = (148.5 - 35, 105 - 35, -1.5)     # carve box in the mat frame (mm): 70 x 70 x 46.5 above the page centre
BOX_HI = (148.5 + 35, 105 + 35, 45.0)

# Answers (slot depth, ridge size, plate thickness) come from each job: measured where the video shows them, otherwise estimated
# (model.estimate_answers) and asked. Part B's caliper values live in backend/tests/fixtures/part_b.json, not here.

# ridge cross-section: a hill (straight sides, rounded peak). Its top radius is an ESTIMATE until a user answers: no frame shows it edge-on.
BUMP_TOP_RADIUS_MM = 0.5
TOP_SHAPE_RADIUS_MM = {"sharp": 0.1, "slightly rounded": 0.5, "quite rounded": 1.0}   # answers of the "top shape" card (estimates)

# "Review your model" anomaly finder (v4 stage 3), tuned on the owner's three marks on the v3 part B model (2304 threshold sets): recall 3/3,
# 0 false alarms, 2 items. Tuned and tested on the same marks of one part: not a held-out result. The placeholders (rms 0.15, min views 3,
# disagreement 1.0, area 4) gave recall 3/3 with 15 false alarms in 24 items. min views 0 = the few-views flag is off (every setting >= 1 added
# false alarms on this part); disagreement 2.0 ties with "off" here (same 2 items) and is kept on.
REVIEW_LUMP_RADIUS_MM = 2.5
REVIEW_LUMP_RMS_MM = 0.4        # top of the tuning grid: a less sensitive value was not tried
REVIEW_MIN_VIEWS = 0
REVIEW_DISAGREE_MM = 2.0
REVIEW_MIN_AREA_MM2 = 16.0
