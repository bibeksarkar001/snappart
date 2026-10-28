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
SAM_CONCEPTS = ("brown plastic buckle", "brown plastic part", "brown object")   # main concept, then fallbacks for frames it misses
MIN_COLOUR_IN_SAM = 0.85
MIN_SAM_CONF = 0.5

# registration (v2 stage 4): Powell refinement voxel. 0.5 mm stays within 0.3 mm of the v1 transform (measured 0.197 mm, 138 s);
# 1.0 mm is faster (29 s) but moved video 2 by 0.304 mm, just over the 0.3 mm limit, so it is not the default.
REG_REFINE_VOXEL_MM = 0.5

# carve (v1 stage 5, held-out IoU)
K_ROBUST = 12                   # a voxel is outside only if 12 views put it outside
GROW_MM = 0.1                   # every silhouette dilated by 0.1 mm
VOXEL_MM = 0.15
TAUBIN_ITERS = 5
BOX_LO = (148.5 - 35, 105 - 35, -1.5)     # carve box in the mat frame (mm): 70 x 70 x 46.5 above the page centre
BOX_HI = (148.5 + 35, 105 + 35, 45.0)

# features (v2 stage 1)
GROOVE_DEPTH_USER_MM = 3.7      # owner's caliper value, NOT measured from the videos

# owner facts: USER-SUPPLIED (calipers), not measured from the videos; editable answers for model.rebuild()
GROOVE_DEPTH_MM = GROOVE_DEPTH_USER_MM
BUMP_LENGTH_USER_MM = 19.7
BUMP_WIDTH_USER_MM = 2.3        # owner also said 2.2 once: to be confirmed
BUMP_HEIGHT_USER_MM = 2.3       # height above the surface the grooves open in
PLATE_THICKNESS_USER_MM = 2.6   # thin flat plate between the window and the groove column (re-measured; replaces 0.9-1.0)
