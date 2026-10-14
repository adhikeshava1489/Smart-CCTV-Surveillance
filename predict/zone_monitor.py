"""
Smart CCTV Surveillance System - Stage 4: Virtual Intrusion & Perimeter Monitor
================================================================================
Monitors user-defined virtual security zones (tripwires / restricted polygons).
Triggers an INTRUSION ALERT whenever a person breaches the security boundary.

Features:
- Configurable normalized polygon zones (works across any camera resolution).
- Real-time Point-in-Polygon intersection check.
- Visual overlay: Semi-transparent Green when secure, pulsing Red upon intrusion.
- Dwell-time tracking (measures how long the intruder has been inside the zone).
"""

import cv2
import numpy as np


class ZoneMonitor:
    """Virtual Security Zone & Tripwire Intrusion Monitor."""
    def __init__(self, zone_polygon=None):
        # Default zone: Central restricted perimeter (normalized [0, 1])
        # Can be customized by passing a list of (x, y) tuples
        if zone_polygon is None:
            self.normalized_polygon = np.array([
                [0.15, 0.35],
                [0.85, 0.35],
                [0.85, 0.90],
                [0.15, 0.90]
            ], dtype=np.float32)
        else:
            poly = np.array(zone_polygon, dtype=np.float32)
            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
                raise ValueError("zone_polygon must be a sequence of at least 3 (x, y) coordinate pairs.")
            self.normalized_polygon = poly

        self.intrusion_count = 0
        self.intrusion_active = False
        self.current_dwell_frames = 0
        self.was_intruding = False

    def get_pixel_polygon(self, width: int, height: int):
        """Converts normalized coordinates to absolute frame pixel coordinates."""
        pts = self.normalized_polygon.copy()
        pts[:, 0] *= width
        pts[:, 1] *= height
        return pts.astype(np.int32)

    def check_intrusion(self, person_boxes, frame_shape):
        """
        Checks if any detected person's foot/center point is inside the restricted zone.
        person_boxes: list of [x1, y1, x2, y2]
        frame_shape: (height, width)
        """
        h, w = frame_shape[:2]
        poly = self.get_pixel_polygon(w, h)

        intrusions = []
        for box in person_boxes:
            x1, y1, x2, y2 = box
            # Use lower center of bounding box (feet point on the ground)
            cx = (x1 + x2) / 2.0
            cy = y2  # ground contact point

            # cv2.pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
            dist = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
            if dist >= 0:
                intrusions.append((box, (int(cx), int(cy))))

        self.intrusion_active = len(intrusions) > 0
        if self.intrusion_active:
            self.current_dwell_frames += 1
            if not self.was_intruding:
                self.intrusion_count += 1
        else:
            self.current_dwell_frames = 0
        self.was_intruding = self.intrusion_active

        return self.intrusion_active, intrusions

    def draw_zone(self, frame, is_active=None):
        """Draws the virtual security zone overlay on the frame."""
        h, w = frame.shape[:2]
        poly = self.get_pixel_polygon(w, h)
        active = self.intrusion_active if is_active is None else is_active

        # Color: Green when secure, Red when breached
        color = (0, 0, 255) if active else (0, 255, 0)
        overlay = frame.copy()

        # Fill semi-transparent zone
        cv2.fillPoly(overlay, [poly], color)
        alpha = 0.25 if active else 0.12
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw boundary line
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

        # Label zone - anchor at top-left bounding coordinate clamped within frame
        label = "RESTRICTED ZONE [BREACHED]" if active else "RESTRICTED ZONE [SECURE]"
        min_x = int(np.min(poly[:, 0]))
        min_y = int(np.min(poly[:, 1]))
        tx = max(10, min(min_x, max(10, w - 280)))
        ty = max(25, min_y - 8)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return frame
