import sys
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Optional

import numpy as np
import pandas as pd
import h5py
from scipy.signal import medfilt
import numpy as np, math
from dataclasses import dataclass
from typing import List


from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from scipy.signal import medfilt


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QTextEdit, QMessageBox,
    QGroupBox, QGridLayout, QCheckBox, QComboBox, QDialog, QScrollArea, QFormLayout, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

# ------------------------- Helpers ------------------------- #

def separate_by_indexes(arr: np.ndarray, cut_indexes: List[int]) -> List[np.ndarray]:
    chunks = []
    start = 0
    for idx in cut_indexes:
        chunks.append(arr[start:idx])
        start = idx
    chunks.append(arr[start:])
    return chunks

@dataclass
class IVTResult:
    velocity: np.ndarray
    fix_cx: List[float]
    fix_cy: List[float]
    fix_dur: List[float]
    labels: List[str]
    fix_x: List[float]
    fix_y: List[float]
    fix_starts: List[float]
    fix_ends: List[float]
    sac_cx: List[float]
    sac_cy: List[float]
    sac_starts: List[float]
    sac_ends: List[float]
    sac_dur: List[float]

def ivt(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, thresh: float,
    kernel: int = 3, theta_px: float = 0.02, tau_ms: float = 75.0,
    min_fix: float = 60.0
) -> IVTResult:
    # ---- 0. median filter ----
    if kernel > 1 and kernel % 2 == 1:
        x = medfilt(x, kernel_size=kernel)
        y = medfilt(y, kernel_size=kernel)

    # ---- 1. velocity threshold classification ----
    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        vel = np.sqrt(dx ** 2 + dy ** 2) / dt

    labels_raw = []
    for v in vel:
        if math.isnan(v): labels_raw.append(-1)             # Blink / no‑track
        elif v < thresh:  labels_raw.append( 1)             # Fixation
        else:             labels_raw.append( 0)             # Saccade
    labels_raw.append(labels_raw[-1] if labels_raw else 0)  # align len

    # ---- helper: contiguous index groups ----
    def groups(arr, val):
        buf, out = [], []
        for i, v in enumerate(arr):
            if v == val: buf.append(i)
            elif buf:    out.append(buf); buf = []
        if buf: out.append(buf)
        return out

    # ---- 2. merge close fixation groups ----
    fix_groups = groups(labels_raw, 1)
    merged, i = [], 0
    while i < len(fix_groups):
        g = fix_groups[i]
        while i + 1 < len(fix_groups):
            g2 = fix_groups[i + 1]
            cx1, cy1 = x[g].mean(),  y[g].mean()
            cx2, cy2 = x[g2].mean(), y[g2].mean()
            dist = np.hypot(cx2 - cx1, cy2 - cy1)
            gap_s  = (g2[0] - g[-1] - 1)
            gap_ms = gap_s * 1000.0 / FS
            if dist <= theta_px and gap_ms <= tau_ms:
                for k in range(g[-1] + 1, g2[0]):  # relabel gap as fixation
                    labels_raw[k] = 1
                g = list(range(g[0], g2[-1] + 1)); i += 1
            else:
                break
        merged.append(g); i += 1
    fix_groups = merged

    # ---- 3. prune too‑short fixations ----
    min_len = int(round(min_fix* FS / 1000.0))
    for g in fix_groups:
        if len(g) < min_len:
            for idx in g: labels_raw[idx] = -1
    fix_groups = groups(labels_raw, 1)
    sac_groups = groups(labels_raw, 0)

    # ---- 4. centroids/durations ----
    def centroids(gr):
        cxs, cys, durs, starts, ends = [], [], [], [], []
        for g in gr:
            start_i, end_i = g[0], g[-1] + 1
            xs = np.concatenate([x[g], [x[end_i]]]) if end_i < len(x) else x[g]
            ys = np.concatenate([y[g], [y[end_i]]]) if end_i < len(y) else y[g]
            cxs.append(xs.mean()); cys.append(ys.mean())
            st, en = t[start_i], t[end_i] if end_i < len(t) else t[-1]
            starts.append(st); ends.append(en); durs.append(en - st)
        return cxs, cys, durs, starts, ends

    fix_cx, fix_cy, fix_dur, fix_st, fix_en = centroids(fix_groups)
    sac_cx, sac_cy, sac_dur, sac_st, sac_en = centroids(sac_groups)

    fxs, fys = [], []
    for g in fix_groups:
        for i in g:
            fxs.append(x[i]); fys.append(y[i])

    labels = ["Blink" if l == -1 else "Fixation" if l == 1 else "Saccade"
              for l in labels_raw]

    return IVTResult(
        velocity=vel,
        fix_cx=fix_cx,  fix_cy=fix_cy,  fix_dur=fix_dur,
        labels=labels,
        fix_x=fxs,      fix_y=fys,
        fix_starts=fix_st,  fix_ends=fix_en,
        sac_cx=sac_cx,  sac_cy=sac_cy,
        sac_starts=sac_st, sac_ends=sac_en, sac_dur=sac_dur
    )





def _sizes_from_duration(fix_dur, scale=10.0, min_s=6.0, max_s=250.0):
    """
    Pretvori trajanja fiksacija (u sekundama) u veličine scatter tačaka.
    scale=10 znači '10 * trajanje'. min_s/max_s ograničavaju veličine.
    """
    fd = np.asarray(fix_dur, dtype=float)
    sizes = scale * np.nan_to_num(fd, nan=0.0, posinf=0.0, neginf=0.0)
    if min_s is not None or max_s is not None:
        lo = min_s if min_s is not None else -np.inf
        hi = max_s if max_s is not None else np.inf
        sizes = np.clip(sizes, lo, hi)
    return sizes



# ------------------------- Metrics functions ------------------------- #

fsb_mapping = {"Fixation": 0, "Saccade": 1, "Blink": 2}

def merge_fsb(l, r):
    if l == 2 or r == 2:
        return 2
    elif l == 0 and r == 0:
        return 0
    else:
        return 1

FS = 60.0  # Hz

def fixation_count(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if fixation_indices.size == 0:
        return 0
    fixation_diffs = np.diff(fixation_indices)
    fixation_diffs = fixation_diffs[fixation_diffs > 1]
    return (len(fixation_diffs) + 1) / norm_factor

def saccade_count(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    saccade_indices = np.where(state == 1)[0]
    if saccade_indices.size == 0:
        return 0
    saccade_diffs = np.diff(saccade_indices)
    saccade_diffs = saccade_diffs[saccade_diffs > 1]
    return (len(saccade_diffs) + 1) / norm_factor

def fixation_total_duration(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    return (len(fixation_indices) / FS) / norm_factor if fixation_indices.size > 0 else 0

def saccade_total_duration(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    saccade_indices = np.where(state == 1)[0]
    return (len(saccade_indices) / FS) / norm_factor if saccade_indices.size > 0 else 0

def fixation_average_duration(xl, xr, yl, yr, state, norm_factor=1):
    total = fixation_total_duration(xl, xr, yl, yr, state)
    count = fixation_count(xl, xr, yl, yr, state)
    return (total / count) / norm_factor if count > 0 else 0

def saccade_average_duration(xl, xr, yl, yr, state, norm_factor=1):
    total = saccade_total_duration(xl, xr, yl, yr, state)
    count = saccade_count(xl, xr, yl, yr, state)
    return (total / count) / norm_factor if count > 0 else 0

def fixation_frequency(xl, xr, yl, yr, state, norm_factor=1):
    total_duration = fixation_total_duration(xl, xr, yl, yr, state)
    count = fixation_count(xl, xr, yl, yr, state)
    return (count / total_duration) / norm_factor if total_duration > 0 else 0

def saccade_frequency(xl, xr, yl, yr, state, norm_factor=1):
    total_duration = saccade_total_duration(xl, xr, yl, yr, state)
    count = saccade_count(xl, xr, yl, yr, state)
    return (count / total_duration) / norm_factor if total_duration > 0 else 0

def active_reading_time(xl, xr, yl, yr, state, norm_factor=1):
    return len(state[state != 2]) / norm_factor

def intersection(x1, x2, x3, x4, y1, y2, y3, y4):
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d:
        xs = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        ys = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        if (min(x1, x2) <= xs <= max(x1, x2) and min(x3, x4) <= xs <= max(x3, x4)):
            return xs, ys

def fixation_intersection_coefficient(xl, xr, yl, yr, state, norm_factor=1):
    ii = 0
    fixation_intersection_numbers = []
    while ii < len(xl) - 1:
        if state[ii] != 0:
            ii += 1
        else:
            x = [(xl[ii] + xr[ii]) / 2]
            y = [(yl[ii] + yr[ii]) / 2]
            while ii < len(xl) - 1 and state[ii] == 0:
                x.append((xl[ii] + xr[ii]) / 2)
                y.append((yl[ii] + yr[ii]) / 2)
                ii += 1
            xs = []
            ys = []
            for i in range(len(x) - 1):
                for j in range(i - 1):
                    result = intersection(x[i], x[i + 1], x[j], x[j + 1],
                                          y[i], y[i + 1], y[j], y[j + 1])
                    if result is not None:
                        xs1, ys1 = result
                        xs.append(xs1)
                        ys.append(ys1)
            fixation_intersection_numbers.append(len(xs))
    return np.nanmean(fixation_intersection_numbers) / norm_factor if fixation_intersection_numbers else 0

def saccade_variability(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    saccade_indices = np.where(state == 1)[0]
    saccade_diffs = np.diff(saccade_indices)
    saccade_diffs = saccade_diffs[saccade_diffs > 1]
    return np.nanstd(saccade_diffs) / norm_factor if len(saccade_diffs) > 0 else 0

def fixation_intersection_std(xl, xr, yl, yr, state, norm_factor=1):
    ii = 0
    fixation_intersection_numbers = []
    while ii < len(xl) - 1:
        if state[ii] != 0:
            ii += 1
        else:
            x = [(xl[ii] + xr[ii]) / 2]
            y = [(yl[ii] + yr[ii]) / 2]
            while ii < len(xl) - 1 and state[ii] == 0:
                x.append((xl[ii] + xr[ii]) / 2)
                y.append((yl[ii] + yr[ii]) / 2)
                ii += 1
            xs = []
            ys = []
            for i in range(len(x) - 1):
                for j in range(i - 1):
                    result = intersection(x[i], x[i + 1], x[j], x[j + 1],
                                          y[i], y[i + 1], y[j], y[j + 1])
                    if result is not None:
                        xs1, ys1 = result
                        xs.append(xs1)
                        ys.append(ys1)
            fixation_intersection_numbers.append(len(xs))
    return np.nanstd(fixation_intersection_numbers) / norm_factor if fixation_intersection_numbers else 0

def horizontal_alteration(xl, xr, yl, yr, state, norm_factor=1):
    ii = 0
    fixation_alteration_numbers = []
    while ii < len(xl) - 1:
        if state[ii] != 0:
            ii += 1
        else:
            x = [(xl[ii] + xr[ii]) / 2]
            while ii < len(xl) - 1 and state[ii] == 0:
                x.append((xl[ii] + xr[ii]) / 2)
                ii += 1
            x = np.array(x)
            x_trend = np.diff(x)
            x_trend_sign = (x_trend > 0).astype(int)
            x_trend_change = np.diff(x_trend_sign)
            fixation_alteration_numbers.append(np.sum(x_trend_change != 0))
    return np.nanmean(fixation_alteration_numbers) / norm_factor if fixation_alteration_numbers else 0

def vertical_alteration(xl, xr, yl, yr, state, norm_factor=1):
    ii = 0
    fixation_alteration_numbers = []
    while ii < len(xl) - 1:
        if state[ii] != 0:
            ii += 1
        else:
            y = [(yl[ii] + yr[ii]) / 2]
            while ii < len(xl) - 1 and state[ii] == 0:
                y.append((yl[ii] + yr[ii]) / 2)
                ii += 1
            y = np.array(y)
            y_trend = np.diff(y)
            y_trend_sign = (y_trend > 0).astype(int)
            y_trend_change = np.diff(y_trend_sign)
            fixation_alteration_numbers.append(np.sum(y_trend_change != 0))
    return np.nanmean(fixation_alteration_numbers) / norm_factor if fixation_alteration_numbers else 0

def estimate_fractal_dimension(x, y):
    max_scaling_factor = 30
    n_samples = 6
    scaling_factors = np.linspace(1, max_scaling_factor, n_samples)
    object_mass = []
    for i in scaling_factors:
        fig = plt.figure(figsize=(1 * i, 1 * i))
        ax = fig.subplots()
        ax.plot(x, y, color='black')
        ax.set_xticks([]); ax.set_yticks([])
        plt.box(False)
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        gray = np.mean(image[:, :, :3], axis=2)
        object_mass.append(np.sum(gray < 0.9 * 255))
        plt.close(fig)
    coeffs = np.polyfit(np.log(scaling_factors), np.log(object_mass), 1)
    return coeffs[0]

def fixation_fractal_dimension_mean(xl, xr, yl, yr, state, norm_factor=1):
    ii = 0
    fractal_dimensions = []
    while ii < len(xl) - 1:
        if state[ii] != 0:
            ii += 1
        else:
            x = [(xl[ii] + xr[ii]) / 2]
            y = [(yl[ii] + yr[ii]) / 2]
            while ii < len(xl) - 1 and state[ii] == 0:
                x.append((xl[ii] + xr[ii]) / 2)
                y.append((yl[ii] + yr[ii]) / 2)
                ii += 1
            x = np.array(x); y = np.array(y)
            if len(x) > 1:
                fractal_dimensions.append(estimate_fractal_dimension(x, y))
    return np.nanmean(fractal_dimensions) / norm_factor if fractal_dimensions else 0

def median_fixation_duration(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    durations = [len(g) / FS for g in groups if len(g) > 0]
    return np.median(durations) / norm_factor if durations else 0

def std_fixation_duration(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    durations = [len(g) / FS for g in groups if len(g) > 0]
    return np.nanstd(durations) / norm_factor if durations else 0

def std_fixation_spatial_span(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    spans = []
    for g in groups:
        if len(g) > 1:
            x_avg = (xl[g] + xr[g]) / 2
            y_avg = (yl[g] + yr[g]) / 2
            dx = np.max(x_avg) - np.min(x_avg)
            dy = np.max(y_avg) - np.min(y_avg)
            spans.append(np.sqrt(dx**2 + dy**2))
    return np.nanstd(spans) / norm_factor if spans else 0

def average_fixation_speed(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    speeds = []
    for g in groups:
        if len(g) > 1:
            x_avg = (xl[g] + xr[g]) / 2
            y_avg = (yl[g] + yr[g]) / 2
            d = np.sqrt(np.diff(x_avg)**2 + np.diff(y_avg)**2)
            speeds.append(np.nanmean(d) * FS)
    return np.nanmean(speeds) / norm_factor if speeds else 0

def std_fixation_speed(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    speeds = []
    for g in groups:
        if len(g) > 1:
            x_avg = (xl[g] + xr[g]) / 2
            y_avg = (yl[g] + yr[g]) / 2
            d = np.sqrt(np.diff(x_avg)**2 + np.diff(y_avg)**2)
            speeds.append(np.nanmean(d) * FS)
    return np.nanstd(speeds) / norm_factor if speeds else 0

def percentage_regressive_fixations(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    regressives = 0
    for g in groups:
        if len(g) > 1:
            x_avg = (xl[g] + xr[g]) / 2
            if np.nanmean(np.diff(x_avg)) < 0:
                regressives += 1
    return (regressives / len(groups)) * 100 if groups else 0

def total_duration_regressive_fixations(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    total_duration = 0
    for g in groups:
        if len(g) > 1:
            x_avg = (xl[g] + xr[g]) / 2
            if np.nanmean(np.diff(x_avg)) < 0:
                total_duration += len(g) / FS
    return total_duration / norm_factor

def average_traversed_fixation_path(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    paths = []
    for g in groups:
        if len(g) > 1:
            x_avg = (xl[g] + xr[g]) / 2
            y_avg = (yl[g] + yr[g]) / 2
            d = np.sqrt(np.diff(x_avg)**2 + np.diff(y_avg)**2)
            paths.append(np.sum(d))
    return np.nanmean(paths) / norm_factor if paths else 0

def std_traversed_fixation_path(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    fixation_indices = np.where(state == 0)[0]
    if len(fixation_indices) == 0:
        return 0
    groups = np.split(fixation_indices, np.where(np.diff(fixation_indices) > 1)[0] + 1)
    paths = []
    for g in groups:
        if len(g) > 1:
            x_avg = (xl[g] + xr[g]) / 2
            y_avg = (yl[g] + yr[g]) / 2
            d = np.sqrt(np.diff(x_avg)**2 + np.diff(y_avg)**2)
            paths.append(np.sum(d))
    return np.nanstd(paths) / norm_factor if paths else 0

def std_saccade_spatial_span(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    saccade_indices = np.where(state == 1)[0]
    if len(saccade_indices) < 2:
        return 0
    spans = []
    for i in range(len(saccade_indices)-1):
        idx1, idx2 = saccade_indices[i], saccade_indices[i+1]
        x1 = (xl[idx1] + xr[idx1]) / 2
        y1 = (yl[idx1] + yr[idx1]) / 2
        x2 = (xl[idx2] + xr[idx2]) / 2
        y2 = (yl[idx2] + yr[idx2]) / 2
        spans.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return np.nanstd(spans) / norm_factor if spans else 0

def std_saccade_duration(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    saccade_indices = np.where(state == 1)[0]
    if len(saccade_indices) == 0:
        return 0
    groups = np.split(saccade_indices, np.where(np.diff(saccade_indices) > 1)[0] + 1)
    durations = [len(g) / FS for g in groups if len(g) > 0]
    return np.nanstd(durations) / norm_factor if durations else 0

def std_saccade_speed(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    saccade_indices = np.where(state == 1)[0]
    if len(saccade_indices) < 2:
        return 0
    speeds = []
    for i in range(len(saccade_indices)-1):
        idx1, idx2 = saccade_indices[i], saccade_indices[i+1]
        dt = (idx2 - idx1) / FS
        if dt > 0:
            x1 = (xl[idx1] + xr[idx1]) / 2
            y1 = (yl[idx1] + yr[idx1]) / 2
            x2 = (xl[idx2] + xr[idx2]) / 2
            y2 = (yl[idx2] + yr[idx2]) / 2
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            speeds.append(distance / dt)
    return np.nanstd(speeds) / norm_factor if speeds else 0

def percentage_regressive_saccades(xl, xr, yl, yr, state, norm_factor=1):
    state = np.array(state)
    saccade_indices = np.where(state == 1)[0]
    if len(saccade_indices) < 2:
        return 0
    regressives = 0
    for i in range(len(saccade_indices)-1):
        idx1, idx2 = saccade_indices[i], saccade_indices[i+1]
        x1 = (xl[idx1] + xr[idx1]) / 2
        x2 = (xl[idx2] + xr[idx2]) / 2
        if x2 < x1:
            regressives += 1
    return (regressives / (len(saccade_indices)-1)) * 100

def total_traversed_path(xl, xr, yl, yr, state, norm_factor=1):
    x_avg = (xl + xr) / 2
    y_avg = (yl + yr) / 2
    d = np.sqrt(np.diff(x_avg)**2 + np.diff(y_avg)**2)
    return np.nansum(d) / norm_factor

# Registry: metric_name -> (func, allow_norm)

WORD_NORMALIZABLE = {
    "fixation_count",
    "saccade_count",
    "fixation_total_duration",
    "saccade_total_duration",
    "active_reading_time",
    "total_traversed_path",
    "total_duration_regressive_fixations",
}


# ------------------------- Metrics registry ------------------------- #

METRICS: Dict[str, Tuple[Callable, bool]] = {
    "fixation_count":  (fixation_count,  "fixation_count"  in WORD_NORMALIZABLE),
    "saccade_count":   (saccade_count,   "saccade_count"   in WORD_NORMALIZABLE),
    "fixation_total_duration": (fixation_total_duration,
                                "fixation_total_duration" in WORD_NORMALIZABLE),
    "saccade_total_duration":  (saccade_total_duration,
                                "saccade_total_duration"  in WORD_NORMALIZABLE),
    "fixation_average_duration": (fixation_average_duration, False),
    "saccade_average_duration":  (saccade_average_duration,  False),
    "fixation_frequency":        (fixation_frequency,        False),
    "saccade_frequency":         (saccade_frequency,         False),
    "active_reading_time":       (active_reading_time,
                                  "active_reading_time"      in WORD_NORMALIZABLE),
    "fixation_intersection_coefficient": (fixation_intersection_coefficient, False),
    "saccade_variability":             (saccade_variability,             False),
    "fixation_intersection_std":       (fixation_intersection_std,       False),
    "horizontal_alteration":           (horizontal_alteration,           False),
    "vertical_alteration":             (vertical_alteration,             False),
    # fraktalna metrika izbačena
    "median_fixation_duration":        (median_fixation_duration,        False),
    "std_fixation_duration":           (std_fixation_duration,           False),
    "std_fixation_spatial_span":       (std_fixation_spatial_span,       False),
    "average_fixation_speed":          (average_fixation_speed,          False),
    "std_fixation_speed":              (std_fixation_speed,              False),
    "percentage_regressive_fixations": (percentage_regressive_fixations, False),
    "total_duration_regressive_fixations": (
        total_duration_regressive_fixations,
        "total_duration_regressive_fixations" in WORD_NORMALIZABLE),
    "average_traversed_fixation_path": (average_traversed_fixation_path, False),
    "std_traversed_fixation_path":     (std_traversed_fixation_path,     False),
    "std_saccade_spatial_span":        (std_saccade_spatial_span,        False),
    "std_saccade_duration":            (std_saccade_duration,            False),
    "std_saccade_speed":               (std_saccade_speed,               False),
    "percentage_regressive_saccades":  (percentage_regressive_saccades,  False),
    "total_traversed_path":            (total_traversed_path,
                                        "total_traversed_path" in WORD_NORMALIZABLE),
}



"""
METRICS: Dict[str, Tuple[Callable, bool]] = {
    "fixation_count": (fixation_count, True),
    "saccade_count": (saccade_count, True),
    "fixation_total_duration": (fixation_total_duration, True),
    "saccade_total_duration": (saccade_total_duration, True),
    "fixation_average_duration": (fixation_average_duration, True),
    "saccade_average_duration": (saccade_average_duration, True),
    "fixation_frequency": (fixation_frequency, True),
    "saccade_frequency": (saccade_frequency, True),
    "active_reading_time": (active_reading_time, True),
    "fixation_intersection_coefficient": (fixation_intersection_coefficient, True),
    "saccade_variability": (saccade_variability, True),
    "fixation_intersection_std": (fixation_intersection_std, True),
    "horizontal_alteration": (horizontal_alteration, True),
    "vertical_alteration": (vertical_alteration, True),
    "median_fixation_duration": (median_fixation_duration, True),
    "std_fixation_duration": (std_fixation_duration, True),
    "std_fixation_spatial_span": (std_fixation_spatial_span, True),
    "average_fixation_speed": (average_fixation_speed, True),
    "std_fixation_speed": (std_fixation_speed, True),
    "percentage_regressive_fixations": (percentage_regressive_fixations, True),
    "total_duration_regressive_fixations": (total_duration_regressive_fixations, True),
    "average_traversed_fixation_path": (average_traversed_fixation_path, True),
    "std_traversed_fixation_path": (std_traversed_fixation_path, True),
    "std_saccade_spatial_span": (std_saccade_spatial_span, True),
    "std_saccade_duration": (std_saccade_duration, True),
    "std_saccade_speed": (std_saccade_speed, True),
    "percentage_regressive_saccades": (percentage_regressive_saccades, True),
    "total_traversed_path": (total_traversed_path, True),
}
"""



# ------------------------- Processor ------------------------- #

class EyeTrackerProcessor:
    def __init__(self, threshold: float = 1.25,
                 kernel_size:int = 3,            # ### NEW
                 theta: float = 0.02,            # ### NEW
                 tau:   float = 75.0,            # ### NEW
                 min_fix: float = 60.0):
        
        
        self.threshold   = threshold
        self.kernel_size = kernel_size     # ### NEW
        self.theta       = theta           # ### NEW
        self.tau         = tau             # ### NEW
        self.min_fix     = min_fix 
        
        #paths
        self.h5_path = ''
        self.xdf_path = ''
        self.xlsx_path = ''
        self.parquet_path = ''
        self.generic_path = ''
        self.csv_path = ''

        self.threshold = threshold

        # raw arrays
        self.left_x = self.left_y = self.right_x = self.right_y = None
        self.left_pupil = self.right_pupil = None
        self.time = None
        self.slide_change = None
        self.num_slides = 0
        self.debug_info = {}

        # segmentation
        self.indexes = None
        self.slide_times: List[np.ndarray] = []
        self.slides_labels: List[str] = []

        self.slide_left_x = []
        self.slide_left_y = []
        self.slide_right_x = []
        self.slide_right_y = []
        self.slide_left_pupil = []
        self.slide_right_pupil = []

        # results
        self.ivt_left: List[IVTResult] = []
        self.ivt_right: List[IVTResult] = []
        self.df_samples = None
        self.df_fix = None
        self.df_sac = None
        self.df_pre_first = None
        self.df_metrics = None

        # progress callback
        self.progress_cb: Optional[Callable[[str], None]] = None

        # ---- MAPIRANJE KOORDINATA ----
        self.coord_mode: str = "unknown"  # postavlja se u loaderu ili autodetect

        # Registry svih mapper funkcija
        self._mappers: Dict[str, Callable[[np.ndarray, np.ndarray, int, int], Tuple[np.ndarray, np.ndarray]]] = {
            "tobii_norm": self._map_tobii_norm,
            "pixel": self._map_passthrough,
            # "xdf_norm01": self._map_xdf_norm01,
            # "csv_custom": self._map_csv_custom,
        }
        
    # ------- COMBINED (L+R)/2 helpers ------------------------------------------

    def _split_into_runs(self, state: np.ndarray, target_val: int):
        """Yield (start, end) indices for contiguous runs where state==target_val."""
        i = 0
        n = len(state)
        while i < n:
            if state[i] != target_val:
                i += 1
                continue
            start = i
            while i < n and state[i] == target_val:
                i += 1
            yield start, i - 1  # inclusive
       
        def _combined_arrays(self, slide_idx: int):
            """Return avg‑x, avg‑y, state arrays *clipped to same length*."""
            lx, rx = self.slide_left_x[slide_idx],  self.slide_right_x[slide_idx]
            ly, ry = self.slide_left_y[slide_idx],  self.slide_right_y[slide_idx]
            label  = self.slides_labels[slide_idx]
            st     = self.df_samples.loc[self.df_samples['Slide'] == label, 'FSB_M'].to_numpy()

            n = min(len(lx), len(rx), len(ly), len(ry), len(st))
            avg_x = (lx[:n] + rx[:n]) / 2.0
            avg_y = (ly[:n] + ry[:n]) / 2.0
            return avg_x, avg_y, st[:n]
    
        def _combined_fixation_centroids(self, avg_x, avg_y, state):
            """Return centroid x,y lists for runs where state==0 (Fixation)."""
            fx, fy = [], []
            for s, e in self._split_into_runs(state, 0):
                fx.append(np.nanmean(avg_x[s:e+1]))
                fy.append(np.nanmean(avg_y[s:e+1]))
            return np.array(fx), np.array(fy)
    

    # ------------- LOADERI ------------- #
    def _load_hdf5(self):
        with h5py.File(self.h5_path, 'r') as f1:
            bino = f1['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']
            self.left_x = np.array(bino['left_gaze_x'])
            self.left_y = np.array(bino['left_gaze_y'])
            self.right_x = np.array(bino['right_gaze_x'])
            self.right_y = np.array(bino['right_gaze_y'])
            self.time = np.array(bino['time'])
            self.left_pupil = np.array(bino['left_pupil_measure1'])
            self.right_pupil = np.array(bino['right_pupil_measure1'])
        self.coord_mode = "tobii_norm"

    def _load_xdf(self):
        raise NotImplementedError('XDF loading not implemented yet.')

    def _load_xlsx(self):
        raise NotImplementedError('XLSX loading not implemented yet.')

    def _load_parquet(self):
        raise NotImplementedError('Parquet loading not implemented yet.')

    def _load_generic_csv_gaze(self):
        raise NotImplementedError('CSV gaze loading not implemented yet.')

    # ------------- PUBLIC API ------------- #
    def load(self):
        if self.h5_path:
            self._load_hdf5()
        elif self.xdf_path:
            self._load_xdf()
        elif self.xlsx_path:
            self._load_xlsx()
        elif self.parquet_path:
            self._load_parquet()
        elif self.generic_path:
            self._load_generic_csv_gaze()
        else:
            raise ValueError('No supported gaze data file provided.')

        if self.coord_mode == "unknown":
            self.coord_mode = self._auto_detect_coord_mode()

        if not self.csv_path:
            raise ValueError('CSV (events) file is required.')
        table = pd.read_csv(self.csv_path)
        if 'image.started' not in table.columns:
            raise ValueError("CSV must contain column 'image.started'")
        self.slide_change = table['image.started'].dropna().to_numpy()
        self.num_slides = int(self.slide_change.shape[0])

        cut_idxs = np.searchsorted(self.time, self.slide_change, side='right')
        self.indexes = cut_idxs.tolist()

        self.slide_times = separate_by_indexes(self.time, self.indexes)

        self.slides_labels = [str(i+1) for i in range(self.num_slides)]
        if len(self.slide_times) > self.num_slides:
            self.slides_labels.append('pitanja')

        self.slide_left_x = separate_by_indexes(self.left_x, self.indexes)
        self.slide_left_y = separate_by_indexes(self.left_y, self.indexes)
        self.slide_right_x = separate_by_indexes(self.right_x, self.indexes)
        self.slide_right_y = separate_by_indexes(self.right_y, self.indexes)
        self.slide_left_pupil = separate_by_indexes(self.left_pupil, self.indexes)
        self.slide_right_pupil = separate_by_indexes(self.right_pupil, self.indexes)

        first_cut_idx = self.indexes[0] if self.indexes else 0
        self.df_pre_first = pd.DataFrame({
            'left_x': self.left_x[:first_cut_idx],
            'left_y': self.left_y[:first_cut_idx],
            'right_x': self.right_x[:first_cut_idx],
            'right_y': self.right_y[:first_cut_idx],
            'left_pupil': self.left_pupil[:first_cut_idx],
            'right_pupil': self.right_pupil[:first_cut_idx],
            'time': self.time[:first_cut_idx]
        })

        self.debug_info = {
            'slide_change_len': self.num_slides,
            'slide_times_chunks': len(self.slide_times),
            'labels_len': len(self.slides_labels)
        }

    def run_ivt(self):
        self.ivt_left.clear(); self.ivt_right.clear()
        for i in range(len(self.slide_times)):
            self.ivt_left.append(ivt(self.slide_left_x[i], self.slide_left_y[i], self.slide_times[i], self.threshold,theta_px=self.theta, tau_ms=self.tau,
                min_fix=self.min_fix,
                kernel=self.kernel_size))
            self.ivt_right.append(ivt(self.slide_right_x[i], self.slide_right_y[i], self.slide_times[i], self.threshold,theta_px=self.theta, tau_ms=self.tau,
                min_fix=self.min_fix,
                kernel=self.kernel_size))

    def build_dfs(self):
        rows = []
        for i in range(len(self.slide_times)):
            n = min(len(self.slide_left_x[i]), len(self.slide_left_y[i]), len(self.slide_right_x[i]),
                    len(self.slide_right_y[i]), len(self.slide_left_pupil[i]), len(self.slide_right_pupil[i]),
                    len(self.slide_times[i]), len(self.ivt_left[i].labels), len(self.ivt_right[i].labels))
            slide_label = self.slides_labels[i] if i < len(self.slides_labels) else str(i)
            for j in range(n):
                rows.append({
                    'Trial': f'{i}', 'Slide': slide_label, 'Time [s]': self.slide_times[i][j],
                    'FSB_L': self.ivt_left[i].labels[j],
                    'FSB_R': self.ivt_right[i].labels[j],
                    'left_x': self.slide_left_x[i][j], 'left_y': self.slide_left_y[i][j],
                    'right_x': self.slide_right_x[i][j], 'right_y': self.slide_right_y[i][j],
                    'left_pupil': self.slide_left_pupil[i][j], 'right_pupil': self.slide_right_pupil[i][j],
                    'Name': os.path.basename(self.h5_path or self.generic_path or self.xdf_path or self.xlsx_path or self.parquet_path)
                })
        self.df_samples = pd.DataFrame(rows)[['Trial','Slide','Time [s]','FSB_L','FSB_R','left_x','left_y','right_x','right_y','left_pupil','right_pupil','Name']]
        # map to ints and merge
        self.df_samples['FSB_L_i'] = self.df_samples['FSB_L'].map(fsb_mapping)
        self.df_samples['FSB_R_i'] = self.df_samples['FSB_R'].map(fsb_mapping)
        self.df_samples['FSB_M'] = [merge_fsb(l, r) for l, r in zip(self.df_samples['FSB_L_i'], self.df_samples['FSB_R_i'])]
     

        # fix & sac dfs unchanged (using left/right IVT results)
        fix_rows = []
        for i in range(len(self.slide_times)):
            slide_label = self.slides_labels[i] if i < len(self.slides_labels) else str(i)
            for j in range(len(self.ivt_left[i].fix_cx)):
                fix_rows.append({'Trial': f'{i}','Slide': slide_label,'Eye':'Left','Centroid Number': j+1,
                                 'Start Time [s]': self.ivt_left[i].fix_starts[j],
                                 'End Time [s]': self.ivt_left[i].fix_ends[j],
                                 'Duration [s]': self.ivt_left[i].fix_dur[j],
                                 'Centroid X': self.ivt_left[i].fix_cx[j],'Centroid Y': self.ivt_left[i].fix_cy[j]})
            for j in range(len(self.ivt_right[i].fix_cx)):
                fix_rows.append({'Trial': f'{i}','Slide': slide_label,'Eye':'Right','Centroid Number': j+1,
                                 'Start Time [s]': self.ivt_right[i].fix_starts[j],
                                 'End Time [s]': self.ivt_right[i].fix_ends[j],
                                 'Duration [s]': self.ivt_right[i].fix_dur[j],
                                 'Centroid X': self.ivt_right[i].fix_cx[j],'Centroid Y': self.ivt_right[i].fix_cy[j]})
        self.df_fix = pd.DataFrame(fix_rows)

        sac_rows = []
        for i in range(len(self.slide_times)):
            slide_label = self.slides_labels[i] if i < len(self.slides_labels) else str(i)
            for j in range(len(self.ivt_left[i].sac_cx)):
                sac_rows.append({'Trial': f'{i}','Slide': slide_label,'Eye':'Left','Centroid Number': j+1,
                                 'Start Time [s]': self.ivt_left[i].sac_starts[j],
                                 'End Time [s]': self.ivt_left[i].sac_ends[j],
                                 'Duration [s]': self.ivt_left[i].sac_dur[j],
                                 'Centroid X': self.ivt_left[i].sac_cx[j],'Centroid Y': self.ivt_left[i].sac_cy[j]})
            for j in range(len(self.ivt_right[i].sac_cx)):
                sac_rows.append({'Trial': f'{i}','Slide': slide_label,'Eye':'Right','Centroid Number': j+1,
                                 'Start Time [s]': self.ivt_right[i].sac_starts[j],
                                 'End Time [s]': self.ivt_right[i].sac_ends[j],
                                 'Duration [s]': self.ivt_right[i].sac_dur[j],
                                 'Centroid X': self.ivt_right[i].sac_cx[j],'Centroid Y': self.ivt_right[i].sac_cy[j]})
        self.df_sac = pd.DataFrame(sac_rows)

    def compute_metrics(self, selected_metrics: Dict[str, bool], norm_flags: Dict[str, bool],
                         norm_excel_path: Optional[str] = None):
        #if not self.has_loaded_raw():
        #    QMessageBox.warning(self, 'Data not loaded', 'Please run LOAD first.')
        #   return
        #if not self.has_processed_dfs():
        #    QMessageBox.warning(self, 'Not processed yet', 'Please run Process ONLY or Process & Export first.')
         #   return
        """Compute metrics per slide. selected_metrics[name]=True/False, norm_flags[name]=True/False"""
        # prepare normalization dict
        norm_map = {}
        if norm_excel_path:
            try:
                df_norm = pd.read_excel(norm_excel_path)
                if 'Slide' not in df_norm.columns or 'norm_factor' not in df_norm.columns:
                    raise ValueError("Excel must have columns 'Slide' and 'norm_factor'")
                for _, r in df_norm.iterrows():
                    norm_map[str(r['Slide'])] = float(r['norm_factor'])
            except Exception as e:
                raise ValueError(f"Error reading normalization file: {e}")

        records = []
        for i in range(len(self.slide_times)):
            slide_label = self.slides_labels[i] if i < len(self.slides_labels) else str(i)
            xl = self.slide_left_x[i]; xr = self.slide_right_x[i]
            yl = self.slide_left_y[i]; yr = self.slide_right_y[i]
            # state merged for this slide:
            mask = self.df_samples['Slide'] == slide_label
            state = self.df_samples.loc[mask, 'FSB_M'].to_numpy()
            row = { 'Name': os.path.basename(self.h5_path or self.generic_path or self.xdf_path or self.xlsx_path or self.parquet_path), 'Trial': str(i), 'Slide': slide_label}
            nf_slide = norm_map.get(slide_label, 1.0)
            for mname, (func, _) in METRICS.items():
                if not selected_metrics.get(mname, False):
                    continue
                use_norm = norm_flags.get(mname, False)
                norm_f = nf_slide if use_norm else 1.0
                try:
                    val = func(xl, xr, yl, yr, state, norm_factor=norm_f)
                except Exception as ex:
                    val = np.nan
                row[mname] = val
            records.append(row)
        self.df_metrics = pd.DataFrame(records)

    def export_excel(self, out_path: str, save_samples=True, save_fix=True, save_sac=True, save_pre=True, save_metrics=True):
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            if save_samples and self.df_samples is not None:
                self.df_samples.to_excel(writer, sheet_name='Samples', index=False)
            if save_fix and self.df_fix is not None:
                self.df_fix.to_excel(writer, sheet_name='Fixation_groups', index=False)
            if save_sac and self.df_sac is not None:
                self.df_sac.to_excel(writer, sheet_name='Saccade_groups', index=False)
            if save_pre and self.df_pre_first is not None:
                self.df_pre_first.to_excel(writer, sheet_name='Pre_First_Slide', index=False)
            if save_metrics and self.df_metrics is not None:
                self.df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

    # ------------- MAPIRANJE KOORDINATA ------------- #
    def _map_tobii_norm(self, x_norm: np.ndarray, y_norm: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        x_img = (x_norm + 1.0) * width / 2.0
        y_img = height - (y_norm + 0.5) * height
        return x_img, y_img

    def _map_passthrough(self, x: np.ndarray, y: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        return x, y

    def _auto_detect_coord_mode(self) -> str:
        try:
            x = np.concatenate([self.left_x, self.right_x])
            y = np.concatenate([self.left_y, self.right_y])
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            ymin, ymax = np.nanmin(y), np.nanmax(y)
        except Exception:
            return "unknown"
        if -1.3 <= xmin <= -0.6 and 0.6 <= xmax <= 1.3:
            return "tobii_norm"
        if 0.0 <= xmin <= 0.05 and 0.95 <= xmax <= 1.1 and 0.0 <= ymin <= 0.05 and 0.95 <= ymax <= 1.1:
            return "xdf_norm01"
        if xmax > 50 and ymax > 50:
            return "pixel"
        return "unknown"

    def map_to_img(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert gaze coordinates to image‑pixel space.

        Tobii returns **(0, 0)** when the gaze point is invalid (blink/no‑track).
        We turn those samples into *NaN* so that matplotlib breaks the line
        instead of drawing a jump to the top‑left corner.
        """

        # --- Remove (0, 0) blink samples ----------------------------------
        if self.coord_mode == "tobii_norm":
            bad = (x_arr == 0) & (y_arr == 0)
            if np.any(bad):
                x_arr = x_arr.copy()
                y_arr = y_arr.copy()
                x_arr[bad] = np.nan
                y_arr[bad] = np.nan

        # --- Project according to coordinate mode ------------------------
        fn = self._mappers.get(self.coord_mode)
        if fn is None:
            return self._map_passthrough(x_arr, y_arr, width, height)
        return fn(x_arr, y_arr, width, height)


    # ------------- CRTANJE ------------- #
    def draw_gaze_over_slides(self, slide_folder: str, save_combined: bool, save_per_slide: bool,
                               include_fixations: bool, linewidth: float = 0.3, dot_size: float = 10.0,
                               out_prefix: str = 'gaze_overlay', out_dir: str = None,
                               combined_limit: int = 10):
        from matplotlib.figure import Figure as AggFigure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg

        def emit(msg: str):
            if self.progress_cb:
                self.progress_cb(msg)

        if out_dir is None:
            out_dir = slide_folder
        images = sorted([p for p in os.listdir(slide_folder) if p.lower().endswith(('.jpg', '.png', '.jpeg'))])
        expected = self.num_slides
        if expected != len(images):
            raise ValueError(f"Num of images ({len(images)}) != num of slides ({expected}).")

        if save_combined and expected > combined_limit:
            emit(f"Too many slides ({expected}) za combined; disabling combined.")
            save_combined = False

        if save_combined:
            fig_all = AggFigure(figsize=(12, 4 * expected))
            canvas_all = FigureCanvasAgg(fig_all)
            axs = fig_all.subplots(expected, 2)
            if expected == 1:
                axs = np.array([axs])

        for i in range(1, len(self.slide_times)):
            emit(f"Plotting slide {i}/{expected}")
            img_path = os.path.join(slide_folder, images[i-1])
            img = mpimg.imread(img_path)
            h, w = img.shape[0], img.shape[1]

            lx = self.slide_left_x[i]; ly = self.slide_left_y[i]
            rx = self.slide_right_x[i]; ry = self.slide_right_y[i]
            lx_img, ly_img = self.map_to_img(lx, ly, w, h)
            rx_img, ry_img = self.map_to_img(rx, ry, w, h)

            if save_combined:
                axL = axs[i-1, 0]; axR = axs[i-1, 1]
                for ax, xdat, ydat, eye_label in [(axL, lx_img, ly_img, 'Left'), (axR, rx_img, ry_img, 'Right')]:
                    ax.imshow(img, aspect='auto', zorder=-1)
                    ax.plot(xdat, ydat, label='Gaze', linewidth=linewidth)
                    
                    if include_fixations:
                        fixres = self.ivt_left[i] if eye_label == 'Left' else self.ivt_right[i]

                        cx, cy = self.map_to_img(np.array(fixres.fix_cx), np.array(fixres.fix_cy), w, h)
                        sizes = _sizes_from_duration(fixres.fix_dur, scale=10.0, min_s=6.0, max_s=250.0)
                        ax.scatter(cx, cy,
                                   s=sizes, zorder=5, c='red',
                                   edgecolors='k', linewidths=0.3,
                                   label='Fixations')
                        #sizes = 10 * np.array(fixres.fix_dur)  # fix_dur je već u sekundama                
                        #x.scatter(cx, cy, s=sizes, zorder=5, c='red', edgecolors='k', linewidths=0.3, label='Fixations')
                
                    ax.set_xlim([0, w]); ax.set_ylim([h, 0])
                    ax.set_title(f'{eye_label} Eye - Slide {i}')
                ax.legend(fontsize='small')

            if save_per_slide:
                from matplotlib.figure import Figure as AggFigureSingle
                from matplotlib.backends.backend_agg import FigureCanvasAgg as CanvasSingle
                fig2 = AggFigureSingle(figsize=(12, 4))
                canvas2 = CanvasSingle(fig2)
                ax1, ax2 = fig2.subplots(1, 2)
                ax_pairs = [(ax1, lx_img, ly_img, 'Left', self.ivt_left[i]), (ax2, rx_img, ry_img, 'Right', self.ivt_right[i])]
                for ax, xdat, ydat, eye_label, fixres in ax_pairs:
                    ax.imshow(img, aspect='auto', zorder=-1)
                    ax.plot(xdat, ydat, label='Gaze', linewidth=linewidth)
                    if include_fixations:
                        fixres = self.ivt_left[i] if eye_label == 'Left' else self.ivt_right[i]

                        cx, cy = self.map_to_img(np.array(fixres.fix_cx), np.array(fixres.fix_cy), w, h)
                        sizes = _sizes_from_duration(fixres.fix_dur, scale=10.0, min_s=6.0, max_s=250.0)
                        ax.scatter(cx, cy,
                                   s=sizes, zorder=5, c='red',
                                   edgecolors='k', linewidths=0.3,
                                   label='Fixations')
                    
                    ax.set_xlim([0, w]); ax.set_ylim([h, 0])
                    ax.set_title(f'{eye_label} Eye - Slide {i}')
                    ax.legend(fontsize='small')
                fig2.suptitle(f'Gaze overlay - Slide {i}')
                fig2.tight_layout(); fig2.subplots_adjust(top=0.9)
                out_file = os.path.join(out_dir, f"{out_prefix}_slide{i}.png")
                fig2.savefig(out_file, dpi=150)

        if save_combined:
            fig_all.suptitle('Gaze plots with fixations for all slides')
            fig_all.tight_layout(); fig_all.subplots_adjust(top=0.95)
            out_file = os.path.join(out_dir, f"{out_prefix}_all.png")
            fig_all.savefig(out_file, dpi=150)
            
# ------------------------- Viewer ------------------------- #

class FixationViewer(QDialog):
    def __init__(self, img_path: str, base_img: np.ndarray, w: int, h: int,
                 left_gaze: Tuple[np.ndarray, np.ndarray], left_fix: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 right_gaze: Tuple[np.ndarray, np.ndarray], right_fix: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 save_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(img_path))
        self.resize(900, 620)

        self.img_path = img_path
        self.img = base_img
        self.w, self.h = w, h
        self.save_dir = save_dir

        self.data = {
            'Left':  {'gaze': left_gaze,  'fix': left_fix },
            'Right': {'gaze': right_gaze, 'fix': right_fix}
            
        }
        self.current_eye = 'Left'
        self.show_fix = True

        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel('Eye:'))
        self.eye_cb = QComboBox(); self.eye_cb.addItems(['Left','Right'])
        top_bar.addWidget(self.eye_cb)
        self.cb_show_fix = QCheckBox('Show fixations'); self.cb_show_fix.setChecked(True)
        top_bar.addWidget(self.cb_show_fix)
        top_bar.addStretch(1)
        self.btn_save = QPushButton('Save PNG')
        top_bar.addWidget(self.btn_save)
        layout.addLayout(top_bar)

        self.canvas = FigureCanvas(Figure(figsize=(8, 5)))
        layout.addWidget(self.canvas)
        self.toolbar = NavToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.ax = self.canvas.figure.add_subplot(111)
        self._redraw()

        self.eye_cb.currentTextChanged.connect(self.change_eye)
        self.cb_show_fix.stateChanged.connect(self.toggle_fix)
        self.btn_save.clicked.connect(self.save_png)

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img, aspect='auto', zorder=-1)
        gaze_x, gaze_y = self.data[self.current_eye]['gaze']
        self.ax.plot(gaze_x, gaze_y, linewidth=0.3, label='Gaze')
        if self.show_fix:
            fx, fy, fix_dur = self.data[self.current_eye]['fix']
            if len(fx) > 0:
                sizes = _sizes_from_duration(fix_dur, scale=10.0, min_s=6.0, max_s=250.0)
                self.ax.scatter(fx, fy, s=sizes, c='red', zorder=5, edgecolors='k', linewidths=0.3, label='Fixations')
        self.ax.set_xlim([0, self.w]); self.ax.set_ylim([self.h, 0])
        self.ax.legend(fontsize='small')
        self.canvas.draw_idle()

    def change_eye(self, txt):
        self.current_eye = txt
        self._redraw()

    def toggle_fix(self, state):
        self.show_fix = (state == Qt.Checked)
        self._redraw()

    def save_png(self):
        base = os.path.splitext(os.path.basename(self.img_path))[0]
        suffix = '_view_fix' if self.show_fix else '_view'
        fname = f"{base}_{self.current_eye}{suffix}.png"
        out_file = os.path.join(self.save_dir, fname)
        self.canvas.figure.savefig(out_file, dpi=200)
        QMessageBox.information(self, 'Saved', f'Saved: {out_file}')

# ------------------------- Threads ------------------------- #

class ProcessThread(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, processor: EyeTrackerProcessor, do_export: bool, out_path: str,
                 save_samples: bool, save_fix: bool, save_sac: bool, save_pre: bool, save_metrics: bool):
        super().__init__()
        self.processor = processor
        self.do_export = do_export
        self.out_path = out_path
        self.save_samples = save_samples
        self.save_fix = save_fix
        self.save_sac = save_sac
        self.save_pre = save_pre
        self.save_metrics = save_metrics

    def run(self):
        try:
            self.progress.emit('Loading data...')
            if not getattr(self.processor, 'slide_times', None) or len(self.processor.slide_times) == 0:
                raise RuntimeError('Raw data not loaded. Click Load first.')
            self.progress.emit('Running IVT...')
            self.processor.run_ivt()
            self.progress.emit('Building DataFrames...')
            self.processor.build_dfs()
            if self.do_export:
                self.progress.emit('Exporting Excel...')
                self.processor.export_excel(self.out_path, self.save_samples, self.save_fix,
                                            self.save_sac, self.save_pre, self.save_metrics)
                self.progress.emit(f'Excel saved to: {self.out_path}')
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e))

class PlotThread(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor: EyeTrackerProcessor, slide_folder: str,
                 save_combined: bool, save_per_slide: bool, include_fix: bool,
                 out_dir: str):
        super().__init__()
        self.processor = processor
        self.slide_folder = slide_folder
        self.save_combined = save_combined
        self.save_per_slide = save_per_slide
        self.include_fix = include_fix
        self.out_dir = out_dir

    def run(self):
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)

            self.progress.emit('Drawing gaze overlays...')
            self.processor.progress_cb = lambda msg: self.progress.emit(msg)
            self.processor.draw_gaze_over_slides(
                slide_folder=self.slide_folder,
                save_combined=self.save_combined,
                save_per_slide=self.save_per_slide,
                include_fixations=self.include_fix,
                out_prefix='gaze_overlay',
                out_dir=self.out_dir
            )
            self.processor.progress_cb = None
            self.finished_ok.emit(self.out_dir)
        except Exception as e:
            self.error.emit(str(e))

# ------------------------- GUI: schemas ------------------------- #

DATA_SCHEMAS: Dict[str, List[Dict[str, object]]] = {
    'Tobii HDF5 + CSV': [
        {'key': 'h5_path',  'label': 'HDF5 (gaze):',   'filter': 'HDF5 files (*.h5 *.hdf5 *.hdf)', 'required': True},
        {'key': 'csv_path', 'label': 'CSV (events):',  'filter': 'CSV files (*.csv)',               'required': True},
    ],
    'XDF (LSL)': [
        {'key': 'xdf_path', 'label': 'XDF (gaze+events):', 'filter': 'XDF files (*.xdf)', 'required': True},
    ],
    'CSV only (gaze)': [
        {'key': 'generic_path', 'label': 'CSV (gaze):',   'filter': 'CSV files (*.csv)', 'required': True},
        {'key': 'csv_path',     'label': 'CSV (events):', 'filter': 'CSV files (*.csv)', 'required': True},
    ],
    'XLSX': [
        {'key': 'xlsx_path', 'label': 'Excel (gaze):', 'filter': 'Excel files (*.xlsx)', 'required': True},
        {'key': 'csv_path',  'label': 'CSV (events):', 'filter': 'CSV files (*.csv)',    'required': True},
    ],
}

KEY_SETTERS: Dict[str, Callable[[EyeTrackerProcessor, str], None]] = {
    'h5_path':      lambda p, v: setattr(p, 'h5_path', v),
    'csv_path':     lambda p, v: setattr(p, 'csv_path', v),
    'xdf_path':     lambda p, v: setattr(p, 'xdf_path', v),
    'xlsx_path':    lambda p, v: setattr(p, 'xlsx_path', v),
    'parquet_path': lambda p, v: setattr(p, 'parquet_path', v),
    'generic_path': lambda p, v: setattr(p, 'generic_path', v),
}

# ------------------------- Main Window ------------------------- #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Eye-Tracker IVT Exporter + Metrics')
        self.resize(1120, 780)

        self.processor: EyeTrackerProcessor = None

        self.schema_file_edits: Dict[str, QLineEdit] = {}
        self.schema_file_buttons: Dict[str, QPushButton] = {}

        self.slide_folder = ''
        self.last_plot_out_dir = ''
        self.data_root_dir = ''

        # metrics UI
        self.metric_checks: Dict[str, QCheckBox] = {}
        self.metric_norm_checks: Dict[str, QCheckBox] = {}
        self.norm_excel_path: str = ''

        self._build_ui()

    def _build_ui(self):
        
        
        central = QWidget(); self.setCentralWidget(central)
        vmain = QVBoxLayout(central)

        # Input box
        file_box = QGroupBox('Input files')
        fb_grid = QGridLayout(file_box)

        fb_grid.addWidget(QLabel('Tip dataset-a:'), 0, 0)
        self.schema_cb = QComboBox(); self.schema_cb.addItems(list(DATA_SCHEMAS.keys()))
        fb_grid.addWidget(self.schema_cb, 0, 1)
        self.schema_cb.currentTextChanged.connect(self.on_schema_changed)

        self.root_label = QLabel('Base data folder:')
        self.root_edit  = QLineEdit(); self.root_edit.setReadOnly(True)
        self.root_btn   = QPushButton('Select Folder...')
        self.root_btn.clicked.connect(self.select_data_root)
        fb_grid.addWidget(self.root_label, 1, 0)
        fb_grid.addWidget(self.root_edit,  1, 1)
        fb_grid.addWidget(self.root_btn,   1, 2)

        self.dynamic_host = QWidget()
        self.dynamic_grid = QGridLayout(self.dynamic_host)
        self.dynamic_grid.setColumnStretch(1, 1)
        fb_grid.addWidget(self.dynamic_host, 2, 0, 1, 3)

        self.slide_label = QLabel('Slides folder:')
        self.slide_edit = QLineEdit(); self.slide_edit.setReadOnly(True)
        self.slide_btn = QPushButton('Select Slides Folder...')
        self.slide_btn.clicked.connect(self.select_slide_folder)
        fb_grid.addWidget(self.slide_label, 3, 0)
        fb_grid.addWidget(self.slide_edit,  3, 1)
        fb_grid.addWidget(self.slide_btn,   3, 2)
        
        self.btn_load = QPushButton('Load')
        self.btn_load.clicked.connect(self.load_data)
        fb_grid.addWidget(self.btn_load, 4, 0, 1, 3)   # ceo red ispod slide-foldera

        # NEW: Clean Data button (opens trimming/cleaning window)
        self.btn_open_clean = QPushButton('Clean Data')
        self.btn_open_clean.setEnabled(False)
        self.btn_open_clean.clicked.connect(lambda: open_clean_dialog(self))
        fb_grid.addWidget(self.btn_open_clean, 5, 0, 1, 3)
   # ceo red ispod slide-foldera

        # Parameters
        thresh_box = QGroupBox('Parameters')
        hth = QHBoxLayout(thresh_box)
        hth.addWidget(QLabel('IVT threshold:'))
        self.thresh_spin = QDoubleSpinBox(); self.thresh_spin.setRange(0.0, 1000.0); self.thresh_spin.setDecimals(2);self.thresh_spin.setSingleStep(0.1); self.thresh_spin.setValue(1.25)
        hth.addWidget(self.thresh_spin); hth.addStretch(1)
        
        hth.addWidget(QLabel('Median k:'))
        self.k_spin = QDoubleSpinBox(); self.k_spin.setRange(1, 11)
        self.k_spin.setDecimals(0); self.k_spin.setValue(3)
        hth.addWidget(self.k_spin)
        
        # --- spatial theta (px / norm jedinica) ---
        hth.addWidget(QLabel('θ (dist):'))
        self.theta_spin = QDoubleSpinBox(); self.theta_spin.setRange(0.0, 1000.0)
        self.theta_spin.setDecimals(3); self.theta_spin.setSingleStep(0.01)
        self.theta_spin.setValue(0.02)
        hth.addWidget(self.theta_spin)

    # --- temporal tau (ms) ---
        hth.addWidget(QLabel('τ (gap ms):'))
        self.tau_spin = QDoubleSpinBox(); self.tau_spin.setRange(0.0, 500.0)
        self.tau_spin.setDecimals(1); self.tau_spin.setSingleStep(5.0)
        self.tau_spin.setValue(75.0)
        hth.addWidget(self.tau_spin)

    # --- minimal fix duration ---
        hth.addWidget(QLabel('Min fix ms:'))
        self.minfix_spin = QDoubleSpinBox(); self.minfix_spin.setRange(0.0, 500.0)
        self.minfix_spin.setDecimals(1); self.minfix_spin.setSingleStep(5.0)
        self.minfix_spin.setValue(60.0)
        hth.addWidget(self.minfix_spin)
        
        

        # Export sheets
        exp_box = QGroupBox('Sheets to export')
        hexp = QHBoxLayout(exp_box)
        self.cb_samples = QCheckBox('Samples'); self.cb_samples.setChecked(True)
        self.cb_fix = QCheckBox('Fixation_groups'); self.cb_fix.setChecked(True)
        self.cb_sac = QCheckBox('Saccade_groups'); self.cb_sac.setChecked(True)
        self.cb_pre = QCheckBox('Pre_First_Slide'); self.cb_pre.setChecked(True)
        self.cb_metrics = QCheckBox('Metrics'); self.cb_metrics.setChecked(True)
        for w in (self.cb_samples, self.cb_fix, self.cb_sac, self.cb_pre, self.cb_metrics): hexp.addWidget(w)
        hexp.addStretch(1)

        # Metrics selection
        metrics_box = QGroupBox('Metrics (default ON). Check "Norm" to normalize.')
        m_layout_outer = QVBoxLayout(metrics_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QGridLayout(inner)
        row = 0
        for name, (func, allow_norm) in METRICS.items():
            cb = QCheckBox(name); cb.setChecked(True)
            self.metric_checks[name] = cb
            form.addWidget(cb, row, 0)
            if allow_norm:
                cbn = QCheckBox('Norm'); cbn.setChecked(False)
                self.metric_norm_checks[name] = cbn
                form.addWidget(cbn, row, 1)
            row += 1
        inner.setLayout(form)
        scroll.setWidget(inner)
        m_layout_outer.addWidget(scroll)

        # Norm excel picker
        hnorm = QHBoxLayout()
        hnorm.addWidget(QLabel('Normalization Excel (Slide,norm_factor):'))
        self.norm_edit = QLineEdit(); self.norm_edit.setReadOnly(True)
        btn_norm = QPushButton('Browse...')
        btn_norm.clicked.connect(self.pick_norm_excel)
        hnorm.addWidget(self.norm_edit); hnorm.addWidget(btn_norm)
        m_layout_outer.addLayout(hnorm)

        # Plot box
        plot_box = QGroupBox('Gaze overlay options')
        hplot = QHBoxLayout(plot_box)
        self.cb_combined = QCheckBox('Save combined figure'); self.cb_combined.setChecked(True)
        self.cb_per_slide = QCheckBox('Save each slide separately'); self.cb_per_slide.setChecked(False)
        self.cb_plot_fix = QCheckBox('Include fixation dots'); self.cb_plot_fix.setChecked(True)
        for w in (self.cb_combined, self.cb_per_slide, self.cb_plot_fix): hplot.addWidget(w)
        hplot.addStretch(1)

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_process = QPushButton('Process ONLY')
        self.btn_export  = QPushButton('Process & Export Excel')
        self.btn_metrics = QPushButton('Compute Metrics')
        self.btn_plot    = QPushButton('Draw Gaze Over Slides')
        self.btn_viewer  = QPushButton('Open Viewer (manual)')
        self.btn_process.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_plot.setEnabled(False)
        self.btn_viewer.setEnabled(False)
        self.btn_metrics.setEnabled(False)
        for b in (self.btn_process, self.btn_export, self.btn_metrics, self.btn_plot, self.btn_viewer):
            btn_box.addWidget(b)
        self.btn_process.clicked.connect(lambda: self.start_process(False))
        self.btn_export.clicked.connect(lambda: self.start_process(True))
        self.btn_metrics.clicked.connect(self.compute_metrics_clicked)
        self.btn_plot.clicked.connect(self.start_plot)
        self.btn_viewer.clicked.connect(self.open_viewer_manual)

        self.log = QTextEdit(); self.log.setReadOnly(True)

        # Layout stacking
        vmain.addWidget(file_box)
        vmain.addWidget(thresh_box)
        vmain.addWidget(exp_box)
        vmain.addWidget(metrics_box)
        vmain.addWidget(plot_box)
        vmain.addLayout(btn_box)
        vmain.addWidget(QLabel('Log:'))
        vmain.addWidget(self.log)

        for w in (self.root_label, self.root_edit, self.root_btn):
            w.hide()
        self.on_schema_changed(self.schema_cb.currentText())

    # ---------------- UI handlers ---------------- #
    def on_schema_changed(self, schema_name: str):
        while self.dynamic_grid.count():
            item = self.dynamic_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.schema_file_edits.clear(); self.schema_file_buttons.clear()

        schema = DATA_SCHEMAS[schema_name]
        row = 0
        for spec in schema:
            lbl = QLabel(spec['label'])
            edit = QLineEdit(); edit.setReadOnly(True)
            btn = QPushButton('Browse...')
            self.dynamic_grid.addWidget(lbl,  row, 0)
            self.dynamic_grid.addWidget(edit, row, 1)
            self.dynamic_grid.addWidget(btn,  row, 2)
            self.schema_file_edits[spec['key']] = edit
            self.schema_file_buttons[spec['key']] = btn
            btn.clicked.connect(lambda _, s=spec: self.pick_file_for_spec(s))
            row += 1
        self.update_button_states()


    def pick_file_for_spec(self, spec: Dict[str, object]):
        dialog_type = spec.get('dialog', 'file')
        if dialog_type == 'dir':
            path = QFileDialog.getExistingDirectory(self, f"Select {spec['label']}", self.data_root_dir or '')
        else:
            flt = spec.get('filter', '')
            start = self.data_root_dir or ''
            path, _ = QFileDialog.getOpenFileName(self, f"Select {spec['label']}", start, flt)
        if not path:
            return
        key = spec['key']
        self.schema_file_edits[key].setText(path)
        self.add_log(f"Selected {spec['label']} {path}")
        self.update_button_states()

    def select_data_root(self):
        path = QFileDialog.getExistingDirectory(self, 'Select base data folder', self.data_root_dir or '')
        if path:
            self.data_root_dir = path
            self.root_edit.setText(path)

    def select_slide_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select slides folder', self.data_root_dir or '')
        if path:
            self.slide_folder = path
            self.slide_edit.setText(path)
            self.update_button_states()

    def pick_norm_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select normalization Excel', self.data_root_dir or '', 'Excel (*.xlsx *.xls)')
        if path:
            self.norm_excel_path = path
            self.norm_edit.setText(path)
            self.add_log(f"Normalization file: {path}")

    def ensure_processor(self, require_slides: bool = True) -> bool:
        if self.processor is None:
            self.processor = EyeTrackerProcessor(
                threshold   = self.thresh_spin.value(),
                kernel_size = int(self.k_spin.value()),     # ### NEW
                theta       = self.theta_spin.value(),      # ### NEW
                tau         = self.tau_spin.value(),        # ### NEW
                min_fix     = self.minfix_spin.value())     # ### NEW
        else:
            self.processor.threshold   = self.thresh_spin.value()
            self.processor.kernel_size = int(self.k_spin.value())   # ### NEW
            self.processor.theta       = self.theta_spin.value()    # ### NEW
            self.processor.tau         = self.tau_spin.value()      # ### NEW
            self.processor.min_fix     = self.minfix_spin.value()   # ### NEW

        schema = DATA_SCHEMAS[self.schema_cb.currentText()]
        for spec in schema:
            key = spec['key']
            req = spec['required']
            path = self.schema_file_edits[key].text().strip()
            if not path and req:
                QMessageBox.warning(self, 'Missing file', f"Select file for: {spec['label']}")
                return False
            if path:
                KEY_SETTERS[key](self.processor, path)

        if require_slides and not self.slide_folder:
            QMessageBox.warning(self, 'Missing slides folder', 'Select the slides images folder (Slides folder).')
            return False

        return True

    def start_process(self, do_export: bool):
        if not self.has_loaded_raw():
            QMessageBox.warning(self, 'Data not loaded', 'Please run LOAD first to build the raw DataFrame.')
            return
        # Require pre-loaded raw data (created via the Load button)
        if not hasattr(self, 'processor') or not getattr(self.processor, 'slide_times', None) or len(self.processor.slide_times) == 0:
            QMessageBox.warning(self, 'Not loaded', 'First click Load to create the RAW DataFrame (trials per slide).')
            return
        if not self.ensure_processor(require_slides=False):
            return
        if not self.processor.csv_path:
            QMessageBox.warning(self, 'Missing CSV', 'CSV (events) file is required in chosen schema.')
            return

        out_path = ''
         

        if do_export:
            default_dir = ''
            for attr in ['h5_path', 'xdf_path', 'xlsx_path', 'generic_path', 'parquet_path']:
                p = getattr(self.processor, attr)
                if p:
                    default_dir = os.path.dirname(p); break
            default_name = 'output.xlsx'
            if self.processor.h5_path:
                default_name = os.path.splitext(os.path.basename(self.processor.h5_path))[0] + '_output.xlsx'
            default_path = os.path.join(default_dir, default_name) if default_dir else default_name
            out_path, _ = QFileDialog.getSaveFileName(self, 'Save Excel', default_path, 'Excel (*.xlsx)')
            if not out_path:
                out_path = default_path
                
        self.thread = ProcessThread(self.processor, do_export, out_path,
                                    self.cb_samples.isChecked(), self.cb_fix.isChecked(),
                                    self.cb_sac.isChecked(), self.cb_pre.isChecked(),
                                    self.cb_metrics.isChecked())
        self.thread.progress.connect(self.add_log)
        self.thread.error.connect(self.on_error)
        self.thread.finished_ok.connect(self.on_finished)
        self.add_log('--- Processing started ---')
        self.thread.start()

    def compute_metrics_clicked(self):
        if self.processor is None or self.processor.df_samples is None:
            QMessageBox.information(self, 'Info', 'First run Process ONLY/Export.')
            return
        # collect selections
        sel = {k: cb.isChecked() for k, cb in self.metric_checks.items()}
        norm = {k: self.metric_norm_checks.get(k, QCheckBox()).isChecked() for k in self.metric_checks.keys()}
        try:
            self.processor.compute_metrics(sel, norm, self.norm_excel_path if any(norm.values()) else None)
            self.add_log('Metrics computed.')
        except Exception as e:
            self.on_error(str(e)); return

        # Ask to save df_metrics
        if self.processor.df_metrics is not None:
            out_path, _ = QFileDialog.getSaveFileName(self, 'Save Metrics Excel', '', 'Excel (*.xlsx)')
            if out_path:
                try:
                    self.processor.df_metrics.to_excel(out_path, index=False)
                    self.add_log(f'Metrics saved to: {out_path}')
                except Exception as e:
                    self.on_error(str(e))

    def start_plot(self):
        if not self.has_loaded_raw():
            QMessageBox.warning(self, 'Data not loaded', 'Please run LOAD first (raw data required).')
            return
        if not self.has_slides_folder():
            QMessageBox.warning(self, 'Slides not selected', 'Please select the slides folder first.')
            return
        if not self.ensure_processor(require_slides=True):
            return
        if self.processor is None or self.processor.df_samples is None:
            QMessageBox.information(self, 'Info', 'You must process data first (Process ONLY or Export).')
            return

        default_dir = ''
        for attr in ['h5_path', 'xdf_path', 'xlsx_path', 'generic_path', 'parquet_path']:
            p = getattr(self.processor, attr)
            if p:
                default_dir = os.path.dirname(p); break
        out_dir = QFileDialog.getExistingDirectory(self, 'Select output folder for gaze images', default_dir)
        if not out_dir:
            out_dir = default_dir or ''
            if not out_dir:
                return
        self.last_plot_out_dir = out_dir

        self.plot_thread = PlotThread(self.processor, self.slide_folder,
                                      self.cb_combined.isChecked(), self.cb_per_slide.isChecked(),
                                      self.cb_plot_fix.isChecked(), out_dir)
        self.plot_thread.progress.connect(self.add_log)
        self.plot_thread.error.connect(self.on_error)
        self.plot_thread.finished_ok.connect(self.on_plot_finished)
        self.add_log('--- Plotting started ---')
        self.plot_thread.start()

    def on_plot_finished(self, out_dir: str):
        self.add_log(f'Gaze overlay saved to: {out_dir}')
       

    def open_viewer_manual(self):
        if not self.has_loaded_raw():
            QMessageBox.warning(self, 'Data not loaded', 'Please run LOAD first (raw data required).')
            return
        if not self.has_slides_folder():
            QMessageBox.warning(self, 'Slides not selected', 'Please select the slides folder first.')
            return
        if not self.ensure_processor(require_slides=True):
            return
        if self.processor is None or self.processor.df_samples is None:
            QMessageBox.information(self, 'Info', 'Process data first.'); return

        dlg = SlidePickDialog(self.processor.num_slides, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        slide_idx = dlg.get_index()
        if slide_idx < 1 or slide_idx > self.processor.num_slides:
            return

        img_list = sorted([p for p in os.listdir(self.slide_folder) if p.lower().endswith(('.jpg','.png','.jpeg'))])
        if len(img_list) != self.processor.num_slides:
            QMessageBox.warning(self, 'Mismatch', 'Number of images != number of slides.'); return
        img_path = os.path.join(self.slide_folder, img_list[slide_idx-1])
        img = mpimg.imread(img_path)
        h, w = img.shape[0], img.shape[1]

        lx = self.processor.slide_left_x[slide_idx]
        ly = self.processor.slide_left_y[slide_idx]
        gxL, gyL = self.processor.map_to_img(lx, ly, w, h)
        fixL = self.processor.ivt_left[slide_idx]
        fxL, fyL = self.processor.map_to_img(np.array(fixL.fix_cx), np.array(fixL.fix_cy), w, h)
        fdurL = np.array(fixL.fix_dur)
        
        rx = self.processor.slide_right_x[slide_idx]
        ry = self.processor.slide_right_y[slide_idx]
        gxR, gyR = self.processor.map_to_img(rx, ry, w, h)
        fixR = self.processor.ivt_right[slide_idx]
        fxR, fyR = self.processor.map_to_img(np.array(fixR.fix_cx), np.array(fixR.fix_cy), w, h)
        fdurR = np.array(fixR.fix_dur)
        
        base_dir = ''
        for attr in ['h5_path', 'xdf_path', 'xlsx_path', 'generic_path', 'parquet_path']:
            p = getattr(self.processor, attr)
            if p:
                base_dir = os.path.dirname(p); break
        viewer = FixationViewer(img_path, img, w, h,
                                 (gxL, gyL), (fxL, fyL, fdurL),
                                 (gxR, gyR), (fxR, fyR, fdurR),
                                 base_dir, self)
        viewer.exec_()

    def add_log(self, txt: str):
        self.log.append(txt); self.log.ensureCursorVisible()

    def on_error(self, msg: str):
        self.add_log(f'ERROR: {msg}')
        QMessageBox.critical(self, 'Error', msg)

    def on_finished(self):
        self.add_log('--- Processing finished ---')
        self.update_button_states()  
        QMessageBox.information(self, 'Done', 'Processing finished.')
        
    
    def load_data(self):
            """
            LOAD-only: loads data and builds RAW DataFrame without IVT.
            - Popunjava self.processor.slide_* nizove pozivom self.processor.load()
            - Pravi self.processor.df_raw sa kolonama:
              ['Trial','Slide','Time [s]','left_x','left_y','right_x','right_y','left_pupil','right_pupil','Name']
            - Stores a reference also in self.df_raw (easy access from GUI)
            """
            try:
                if not self.ensure_processor(require_slides=False):
                    return
                if not self.processor.csv_path:
                    QMessageBox.warning(self, 'Missing CSV', 'CSV (events) file is required in chosen schema.')
                    return
                # Učitaj ulazne podatke i podeli po slajdovima
                self.processor.load()
                # Sastavi RAW DF bez IVT labela
                rows = []
                num_slides = len(getattr(self.processor, 'slide_times', []))
                for i in range(num_slides):
                    # dužina po slajdu (min preko svih polja da izbegnemo out-of-range)
                    n = min(len(self.processor.slide_times[i]),
                            len(self.processor.slide_left_x[i]), len(self.processor.slide_left_y[i]),
                            len(self.processor.slide_right_x[i]), len(self.processor.slide_right_y[i]),
                            len(self.processor.slide_left_pupil[i]), len(self.processor.slide_right_pupil[i]))
                    slide_label = self.processor.slides_labels[i] if i < len(getattr(self.processor, 'slides_labels', [])) else str(i+1)
                    base_name = os.path.basename(self.processor.h5_path or self.processor.xdf_path or self.processor.xlsx_path or self.processor.parquet_path or self.processor.generic_path or self.processor.csv_path or '')
                    for j in range(n):
                        rows.append({
                            'Trial': f'{i}', 'Slide': slide_label, 'Time [s]': self.processor.slide_times[i][j],
                            'left_x': self.processor.slide_left_x[i][j], 'left_y': self.processor.slide_left_y[i][j],
                            'right_x': self.processor.slide_right_x[i][j], 'right_y': self.processor.slide_right_y[i][j],
                            'left_pupil': self.processor.slide_left_pupil[i][j], 'right_pupil': self.processor.slide_right_pupil[i][j],
                            'Name': base_name
                        })
                self.processor.df_raw = pd.DataFrame(rows)[['Trial','Slide','Time [s]','left_x','left_y','right_x','right_y','left_pupil','right_pupil','Name']]
                # Referenca i na MainWindow nivou (po želji)
                self.df_raw = self.processor.df_raw
                # Obavesti korisnika
                n_rows = len(self.processor.df_raw)
                n_slides = num_slides
                QMessageBox.information(self, 'Load', f'RAW ready without IVT. Trials: {n_slides}, rows: {n_rows}.')
                self.update_button_states()
                try:
                    self.btn_open_clean.setEnabled(True)
                except Exception:
                    pass
            except Exception as e:
                QMessageBox.critical(self, 'Error (LOAD)', str(e))
    def has_loaded_raw(self) -> bool:
            
        # Važi bilo da si DF čuvao na self.df_raw ili self.processor.df_raw
        df1 = getattr(self, 'df_raw', None)
        df2 = getattr(getattr(self, 'processor', None), 'df_raw', None)
        return df1 is not None or df2 is not None
        
    def has_slides_folder(self) -> bool:
        return bool(getattr(self, 'slide_folder', ''))
        
    
    def _inputs_ready(self) -> bool:
        schema = DATA_SCHEMAS[self.schema_cb.currentText()]
        for spec in schema:
            if spec.get('required', False):
                edit = self.schema_file_edits.get(spec['key'])
                if not edit or not edit.text().strip():
                    return False
        return True

    def _raw_ready(self) -> bool:
        return self.has_loaded_raw()

    def _processed_ready(self) -> bool:
        return self.has_processed_dfs()
    def update_button_states(self):

        has_inputs = self._inputs_ready()
        has_raw    = self._raw_ready()
        has_proc   = self._processed_ready()
        has_slides = self.has_slides_folder()

        # Load enabled only when all required inputs are selected
        try:
            self.btn_load.setEnabled(has_inputs)
        except Exception:
            pass

        # Clean Data, Process, Export after RAW is loaded
        try:
            self.btn_open_clean.setEnabled(has_raw)
        except Exception:
            pass
        self.btn_process.setEnabled(has_raw)
        self.btn_export.setEnabled(has_raw)

        # Metrics after processed
        self.btn_metrics.setEnabled(has_proc)

        # Overlays / Viewer require processed + slides folder
        self.btn_plot.setEnabled(has_proc and has_slides)
        self.btn_viewer.setEnabled(has_proc and has_slides)

    def has_processed_dfs(self) -> bool:
    # standardno posle build_dfs: processor.df_samples postoji
        proc = getattr(self, 'processor', None)
        df = getattr(proc, 'df_samples', None)
        # ako ti je ime drugačije (npr. samples_df), dopuni još jednu proveru:
        if df is None:
                df = getattr(proc, 'samples_df', None)
        return isinstance(df, pd.DataFrame) and not df.empty


# ------------------------- Slide picker ------------------------- #

class SlidePickDialog(QDialog):
    def __init__(self, max_slides: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Pick slide index')
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        h.addWidget(QLabel('Slide (1..{}):'.format(max_slides)))
        self.cb = QComboBox()
        for i in range(1, max_slides+1):
            self.cb.addItem(str(i))
        h.addWidget(self.cb)
        v.addLayout(h)
        btns = QHBoxLayout()
        ok = QPushButton('OK'); ca = QPushButton('Cancel')
        btns.addWidget(ok); btns.addWidget(ca)
        v.addLayout(btns)
        ok.clicked.connect(self.accept); ca.clicked.connect(self.reject)
    def get_index(self) -> int:
        return int(self.cb.currentText())

# ------------------------- main ------------------------- #

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


# --------------- Clean Data Dialog (LOAD-based) ---------------

class MarkerSlider(QtWidgets.QSlider):
    """QSlider sa zelenim/plavim markerima i tankim crvenim crtama za NaN."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.start_marker = None
        self.end_marker = None
        self.missing_indices = []

    def setStartMarker(self, value: int):
        self.start_marker = int(value) if value is not None else None
        self.update()

    def setEndMarker(self, value: int):
        self.end_marker = int(value) if value is not None else None
        self.update()

    def setMissingIndices(self, indices):
        try:
            self.missing_indices = [int(i) for i in indices]
        except Exception:
            self.missing_indices = []
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        min_val = self.minimum()
        max_val = self.maximum()
        span = max(1, max_val - min_val)
        # Start marker (green)
        if self.start_marker is not None:
            x = (self.start_marker - min_val) / span * (self.width() - 1)
            yb = self.height() - 2
            pts = [QtCore.QPointF(x-5, yb), QtCore.QPointF(x+5, yb), QtCore.QPointF(x, yb-10)]
            painter.setBrush(QtGui.QColor('green')); painter.setPen(QtGui.QColor('green'))
            painter.drawPolygon(QtGui.QPolygonF(pts))
        # End marker (blue)
        if self.end_marker is not None:
            x = (self.end_marker - min_val) / span * (self.width() - 1)
            yb = self.height() - 2
            pts = [QtCore.QPointF(x-5, yb), QtCore.QPointF(x+5, yb), QtCore.QPointF(x, yb-10)]
            painter.setBrush(QtGui.QColor('blue')); painter.setPen(QtGui.QColor('blue'))
            painter.drawPolygon(QtGui.QPolygonF(pts))
        # Missing ticks (red)
        if self.missing_indices:
            pen = QtGui.QPen(QtGui.QColor('red')); pen.setWidth(1); painter.setPen(pen)
            for idx in self.missing_indices:
                if idx < min_val or idx > max_val: continue
                x = (idx - min_val) / span * (self.width() - 1)
                painter.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, 5))
        painter.end()


class CleanDataDialog(QtWidgets.QDialog):
    """Interaktivno seckanje po slajdu nad df_raw/slide_* iz LOAD-a."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Ukloni '?' (context help) iz naslovne linije
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self._last_pos = {}
        self.setWindowTitle("Clean Data — Trim trials")
        self.resize(1200, 800)
        self.parent_main = parent

        # Dohvati df_raw (kreiran u Load) i grupiši po Trial
        self.df_raw = _get_df_raw_from(parent)
        if not isinstance(self.df_raw, pd.DataFrame) or self.df_raw.empty or 'Trial' not in self.df_raw.columns:
            QtWidgets.QMessageBox.warning(self, "No data", "Load the data first")
            self.reject()
            return

        # Napravi listu dostupnih trialova (po redu pojave)
        self.trials = [t for t in sorted(self.df_raw['Trial'].astype(int).unique()) if t != 0]
        self.current_trial = self.trials[0] if self.trials else None

        # Raspored: levo kontrole, desno prikaz + slider
        root = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        lbl = QtWidgets.QLabel("Trial selection")
        self.combo = QtWidgets.QComboBox()
        for t in self.trials:
            # Ako postoji kolona 'Slide' (labela), prikazi i nju
            label = None
            try:
                label = self.df_raw.loc[self.df_raw['Trial']==t, 'Slide'].dropna().iloc[0]
            except Exception:
                label = str(t+1)
            self.combo.addItem(f"Trial {t}", userData=t)
        self.btn_set_start = QtWidgets.QPushButton("Trial start")
        self.btn_set_end = QtWidgets.QPushButton("Trial end")
        self.lbl_start = QtWidgets.QLabel("Start: /")
        self.lbl_end = QtWidgets.QLabel("End: /")
        left.addWidget(lbl); left.addWidget(self.combo)
        left.addWidget(self.btn_set_start); left.addWidget(self.btn_set_end)
        self._help = QtWidgets.QLabel('Default Start is the beginning of the trial. Default End is the end of the trial. By moving the slider to the deisred postition and clicking Trial start/end the postion of markers is adjusted.')
        
        self._help.setWordWrap(True)
        left.addWidget(self._help)
        left.addWidget(self.lbl_start); left.addWidget(self.lbl_end)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        # Nav dugmad
        nav = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Previous trial")
        self.next_btn = QtWidgets.QPushButton("Next trial")
        nav.addWidget(self.prev_btn); nav.addStretch(1); nav.addWidget(self.next_btn)
        right.addLayout(nav)

        # Scena i prikaz
        self.scene = QtWidgets.QGraphicsScene(); self.scene.setSceneRect(0,0,1920,1080)
        self.view = QtWidgets.QGraphicsView(); self.view.setScene(self.scene)
        self.view.setSceneRect(self.scene.sceneRect())
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        right.addWidget(self.view, 1)

        # Slider
        self.slider = MarkerSlider(QtCore.Qt.Horizontal); self.slider.setMaximum(100)
        right.addWidget(self.slider)

        root.addLayout(left); root.addLayout(right, 1)

        # Stanje
        self.path_item = None
        self.scaled_x = np.array([]); self.scaled_y = np.array([])
        self.ranges = {}  # trial -> (s,e)

        # Signali
        self.combo.currentIndexChanged.connect(self.on_select_trial)
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.btn_set_start.clicked.connect(self.on_set_start)
        self.btn_set_end.clicked.connect(self.on_set_end)

        # Dugme OK/Cancel
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.apply_and_close)
        btn_box.rejected.connect(self.reject)
        right.addWidget(btn_box)

        # Init
        if self.current_trial is not None:
            self._load_current_trial()

    # --- Navigacija ---
    def on_prev(self):
        idx = self.combo.currentIndex()
        if idx > 0: self.combo.setCurrentIndex(idx - 1)

    def on_next(self):
        idx = self.combo.currentIndex()
        if idx < self.combo.count() - 1: self.combo.setCurrentIndex(idx + 1)

    def on_select_trial(self, idx: int):
        if idx < 0: return
        trial = int(self.combo.itemData(idx))
        self.current_trial = trial
        self._load_current_trial()

    # --- Učitavanje i crtanje ---
    def _load_current_trial(self):
        self.scene.clear(); self.path_item = None
        df = self.df_raw[self.df_raw['Trial'].astype(int) == int(self.current_trial)].copy()
        # resetuj indeks radi lakse manipulacije po uzorku
        df.reset_index(drop=True, inplace=True)

        # Odredi NaN pozicije (ako bilo koji od koord. je NaN)
        nan_mask = (
            df[['left_x','left_y','right_x','right_y']].isna().any(axis=1)
            if set(['left_x','left_y','right_x','right_y']).issubset(df.columns)
            else pd.Series(False, index=df.index)
        )
        missing_indices = np.where(nan_mask.to_numpy())[0].tolist()
        self.slider.setMissingIndices(missing_indices)

        # Mean(L/R)
        try:
            x = ((df['left_x'].to_numpy(float) + df['right_x'].to_numpy(float)) / 2.0)
            y = ((df['left_y'].to_numpy(float) + df['right_y'].to_numpy(float)) / 2.0)
        except Exception:
            x = df.get('left_x', pd.Series(dtype=float)).to_numpy(float)
            y = df.get('left_y', pd.Series(dtype=float)).to_numpy(float)

        # Skaliranje u 1920x1080 uz 90p
        if x.size and y.size:
            y_inv = -y
            xc = np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0
            yc = np.nanmedian(y_inv) if np.isfinite(np.nanmedian(y_inv)) else 0.0
            dx = np.abs(x - xc); dy = np.abs(y_inv - yc)
            try: xl = np.nanpercentile(dx, 90)
            except Exception: xl = np.nanmax(dx) if dx.size else 1.0
            try: yl = np.nanpercentile(dy, 90)
            except Exception: yl = np.nanmax(dy) if dy.size else 1.0
            xl = 1.0 if not np.isfinite(xl) or xl == 0 else xl
            yl = 1.0 if not np.isfinite(yl) or yl == 0 else yl
            sx = 1920.0 / (2.0 * xl); sy = 1080.0 / (2.0 * yl)
            offx, offy = 960.0, 540.0
            self.scaled_x = np.clip(offx + (x - xc) * sx * 0.5, 0, 1920)
            self.scaled_y = np.clip(offy + (y_inv - yc) * sy * 0.5, 0, 1080)
        else:
            self.scaled_x = np.array([]); self.scaled_y = np.array([])

        n = len(self.scaled_x)
        self.slider.setMaximum(max(n - 1, 1))
        # Primeni postojece markere ako vec postoje
        s, e = self.ranges.get(self.current_trial, (0, 0))
        self.lbl_start.setText(f"Start: {s}")
        self.lbl_end.setText(f"End: {e}")
        self.slider.setStartMarker(s); self.slider.setEndMarker(e)

        # Podesi vrednost slidera: ako trial nije ranije pomeran → 0, inače poslednja pozicija
        try:
            pos = int(self._last_pos.get(self.current_trial, 0))
            pos = max(0, min(pos, self.slider.maximum()))
            self.slider.setValue(pos)
        except Exception:
            self.slider.setValue(0)

        self.on_slider_change()

    def on_slider_change(self):
        v = self.slider.value()
        
        # zapamti poslednju poziciju slidera za trenutni trial
        try:
            self._last_pos[self.current_trial] = int(v)
        except Exception:
            pass
        if not self.scaled_x.size: return
        cx = self.scaled_x[:v+1]; cy = self.scaled_y[:v+1]
        if self.path_item is None:
            self.path_item = QtWidgets.QGraphicsPathItem()
            self.path_item.setPen(QtGui.QPen(QtGui.QColor('black'), 1.5))
            self.scene.addItem(self.path_item)
        path = QtGui.QPainterPath()
        path.moveTo(QtCore.QPointF(cx[0], cy[0]))
        for x, y in zip(cx[1:], cy[1:]):
            path.lineTo(QtCore.QPointF(x, y))
        self.path_item.setPath(path)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    # --- Markeri ---
    def on_set_start(self):
        v = int(self.slider.value())
        self.ranges[self.current_trial] = (v, self.ranges.get(self.current_trial, (v, v))[1])
        self.lbl_start.setText(f"Start: {v}")
        self.slider.setStartMarker(v)

    def on_set_end(self):
        v = int(self.slider.value())
        self.ranges[self.current_trial] = (self.ranges.get(self.current_trial, (0, v))[0], v)
        self.lbl_end.setText(f"End: {v}")
        self.slider.setEndMarker(v)

    # --- Primena ---
    def apply_and_close(self):
        if not self.ranges:
            self.accept(); return

        # Dodaj per-trial indeks da bi se seckalo po uzorku
        df = self.df_raw.copy()
        df['_idx'] = df.groupby('Trial').cumcount()

        # Primeni seckanje po svakom trialu
        keep_list = []
        for t, (s, e) in self.ranges.items():
            s = int(max(0, s)); e = int(max(0, e))
            if e < s: s, e = e, s
            mask = (df['Trial'].astype(int) == int(t)) & (df['_idx'].between(s, e, inclusive='both'))
            keep_list.append(mask)
        if keep_list:
            keep = keep_list[0]
            for m in keep_list[1:]:
                keep |= m
            df_trim = df[keep].drop(columns=['_idx']).reset_index(drop=True)
        else:
            df_trim = df.drop(columns=['_idx']).reset_index(drop=True)

        # Upisi nazad u parent
        try:
            self.parent_main.df_raw = df_trim
            if hasattr(self.parent_main, 'processor'):
                self.parent_main.processor.df_raw = df_trim
                # Ako postoje slide_* nizovi, iseći i njih po istim granicama
                slides = len(getattr(self.parent_main.processor, 'slide_times', []))
                for t, (s, e) in self.ranges.items():
                    i = int(t)
                    if i < 0 or i >= slides: continue
                    # sigurnosno klampovanje
                    def _slice(arr):
                        try:
                            n = len(arr[i])
                            ss = max(0, min(int(s), n-1))
                            ee = max(0, min(int(e), n-1))
                            if ee < ss: ss, ee = ee, ss
                            arr[i] = arr[i][ss:ee+1]
                        except Exception:
                            pass
                    for name in ['slide_times','slide_left_x','slide_left_y','slide_right_x','slide_right_y','slide_left_pupil','slide_right_pupil']:
                        if hasattr(self.parent_main.processor, name):
                            _slice(getattr(self.parent_main.processor, name))
        except Exception:
            pass

        self.accept()



def _get_df_raw_from(main_window):
    """Return df_raw from main_window or its processor without using Python 'or' on DataFrames."""
    try:
        df1 = getattr(main_window, 'df_raw', None)
        if isinstance(df1, pd.DataFrame):
            return df1
    except Exception:
        pass
    try:
        proc = getattr(main_window, 'processor', None)
        if proc is not None:
            df2 = getattr(proc, 'df_raw', None)
            if isinstance(df2, pd.DataFrame):
                return df2
    except Exception:
        pass
    return None

def open_clean_dialog(main_window):
    df_raw = _get_df_raw_from(main_window)
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        QtWidgets.QMessageBox.warning(main_window, "No RAW data", "Click Load first, then Clean Data.")
        return
    dlg = CleanDataDialog(main_window)
    dlg.exec_()


if __name__ == '__main__':
    main()
