"""Precompute sliding-window HGA animation frames for the Insula HGA Explorer."""

from __future__ import annotations

import math
from typing import Any

PHASES = ("stimulus", "delay", "go", "response")
V1_TASKS = ("PhonemeSequencing", "LexicalDelay")
DEFAULT_CONDITION = "Repeat"
CONDITIONS_BY_TASK = {
    "PhonemeSequencing": ["Repeat"],
    "LexicalDelay": ["Repeat", "Decision"],
}
DISPLAY_WAVEFORM_RANGE = (-0.5, 1.0)

ANIM_WINDOW_SEC = 0.2
ANIM_STEP_SEC = 0.02
ANIM_GAUSSIAN_SIGMA_SEC = 0.04


def selection_key(task: str, condition: str) -> str:
    return f"{task}|{condition}"


def animation_bundle_keys(tasks: list[str] | None = None) -> tuple[str, ...]:
    task_list = list(tasks or V1_TASKS)
    keys = [selection_key("all", DEFAULT_CONDITION)]
    for task in task_list:
        for condition in CONDITIONS_BY_TASK.get(task, [DEFAULT_CONDITION]):
            keys.append(selection_key(task, condition))
    return tuple(dict.fromkeys(keys))


def interpolate_trace_value(trace: dict[str, list], time: float) -> float | None:
    times = trace.get("time") or []
    values = trace.get("value") or []
    if not times:
        return None
    if time <= times[0]:
        value = values[0]
        return None if value is None else float(value)
    if time >= times[-1]:
        value = values[-1]
        return None if value is None else float(value)
    for index in range(len(times) - 1):
        if times[index] <= time <= times[index + 1]:
            span = times[index + 1] - times[index]
            if span == 0:
                value = values[index]
                return None if value is None else float(value)
            weight = (time - times[index]) / span
            left = values[index]
            right = values[index + 1]
            if left is None or right is None:
                return None
            return float(left + weight * (right - left))
    return None


def average_traces(traces: list[dict[str, list]]) -> dict[str, list] | None:
    if not traces:
        return None
    if len(traces) == 1:
        trace = traces[0]
        return {"time": trace["time"], "value": trace["value"]}
    time_set = set()
    for trace in traces:
        time_set.update(trace["time"])
    times = sorted(time_set)
    values = []
    for time in times:
        samples = [interpolate_trace_value(trace, time) for trace in traces]
        samples = [value for value in samples if value is not None]
        values.append(sum(samples) / len(samples) if samples else None)
    return {"time": times, "value": values}


def resolve_phase_trace(
    traces: dict,
    electrode_id: str,
    task: str,
    phase: str,
    condition: str,
    tasks: list[str] | None = None,
) -> dict[str, list] | None:
    electrode_traces = traces.get(electrode_id, {})
    if task == "all":
        candidates = []
        for task_name in tasks or V1_TASKS:
            trace = electrode_traces.get(task_name, {}).get(phase, {}).get(condition)
            if trace:
                candidates.append(trace)
        return average_traces(candidates)
    return electrode_traces.get(task, {}).get(phase, {}).get(condition)


def parse_selection_key(selected_key: str) -> tuple[str, str]:
    if "|" not in selected_key:
        return selected_key, DEFAULT_CONDITION
    task, condition = selected_key.split("|", 1)
    return task, condition


def window_mean(trace: dict[str, list], t0: float, t1: float) -> float | None:
    if not trace or t1 <= t0:
        return None
    n_samples = max(4, math.ceil((t1 - t0) / 0.015625))
    samples = []
    for index in range(n_samples + 1):
        t = t0 + (index / n_samples) * (t1 - t0)
        value = interpolate_trace_value(trace, t)
        if value is not None and math.isfinite(value):
            samples.append(value)
    if not samples:
        return None
    return sum(samples) / len(samples)


def causal_window_mean_for_electrode(
    traces: dict,
    electrode_id: str,
    phase: str,
    selected_key: str,
    time: float,
    window_sec: float,
    tasks: list[str] | None = None,
) -> float | None:
    task, condition = parse_selection_key(selected_key)
    trace = resolve_phase_trace(traces, electrode_id, task, phase, condition, tasks=tasks)
    if not trace:
        return None
    return window_mean(trace, time, time + window_sec)


def causal_gaussian_smooth_series(
    times: list[float],
    values: list[float | None],
    sigma_sec: float,
) -> list[float | None]:
    if not values:
        return []
    if sigma_sec <= 0:
        return list(values)
    sigma_sq2 = 2 * sigma_sec * sigma_sec
    smoothed = []
    for index, _ in enumerate(values):
        t_current = times[index]
        weighted_sum = 0.0
        weight_sum = 0.0
        for j in range(index + 1):
            value = values[j]
            if value is None or not math.isfinite(value):
                continue
            dt = t_current - times[j]
            weight = math.exp(-(dt * dt) / sigma_sq2)
            weighted_sum += weight * value
            weight_sum += weight
        smoothed.append(weighted_sum / weight_sum if weight_sum > 0 else None)
    return smoothed


def smooth_animation_frames(
    frames: list[dict],
    sigma_sec: float = ANIM_GAUSSIAN_SIGMA_SEC,
) -> list[dict]:
    if not frames or sigma_sec <= 0:
        return frames
    times = [frame["time"] for frame in frames]
    electrode_ids = sorted({
        electrode_id
        for frame in frames
        for electrode_id in frame["hgaByElectrodeId"]
    })
    smoothed_by_electrode = {}
    for electrode_id in electrode_ids:
        values = [frame["hgaByElectrodeId"].get(electrode_id) for frame in frames]
        smoothed_by_electrode[electrode_id] = causal_gaussian_smooth_series(times, values, sigma_sec)
    smoothed_frames = []
    for index, frame in enumerate(frames):
        hga_by_electrode_id = {}
        for electrode_id in electrode_ids:
            value = smoothed_by_electrode[electrode_id][index]
            if value is not None and math.isfinite(value):
                hga_by_electrode_id[electrode_id] = value
        smoothed_frames.append({"time": frame["time"], "hgaByElectrodeId": hga_by_electrode_id})
    return smoothed_frames


def percentile95(values: list[float]) -> float:
    if not values:
        return 1.0
    sorted_values = sorted(abs(value) for value in values)
    index = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return sorted_values[index] if sorted_values[index] > 0 else 1.0


def build_sliding_window_frames(
    electrode_ids: list[str],
    traces: dict,
    phase: str,
    selected_key: str,
    tasks: list[str] | None = None,
    window_sec: float = ANIM_WINDOW_SEC,
    step_sec: float = ANIM_STEP_SEC,
    display_range: tuple[float, float] = DISPLAY_WAVEFORM_RANGE,
) -> dict[str, Any]:
    task, condition = parse_selection_key(selected_key)
    phase_min, phase_max = display_range
    t_end = phase_max - window_sec
    frames = []
    t = phase_min
    while t <= t_end + 1e-9:
        hga_by_electrode_id = {}
        for electrode_id in electrode_ids:
            mean = causal_window_mean_for_electrode(
                traces,
                electrode_id,
                phase,
                selected_key,
                t,
                window_sec,
                tasks=tasks,
            )
            if mean is not None and math.isfinite(mean):
                hga_by_electrode_id[electrode_id] = mean
        frames.append({"time": round(t, 4), "hgaByElectrodeId": hga_by_electrode_id})
        t += step_sec

    smoothed_frames = smooth_animation_frames(frames)
    smoothed_values = [
        value
        for frame in smoothed_frames
        for value in frame["hgaByElectrodeId"].values()
    ]
    active_ids = sorted({
        electrode_id
        for frame in smoothed_frames
        for electrode_id in frame["hgaByElectrodeId"]
    })
    values_matrix = [
        [frame["hgaByElectrodeId"].get(electrode_id) for electrode_id in active_ids]
        for frame in smoothed_frames
    ]
    return {
        "phase": phase,
        "selected_task": task,
        "selected_condition": condition,
        "selection_key": selected_key,
        "times": [frame["time"] for frame in smoothed_frames],
        "electrode_ids": active_ids,
        "values": values_matrix,
        "scale": {
            "vmin": 0.0,
            "vmax": percentile95(smoothed_values),
            "method": "p95_abs_sliding_window_gaussian",
        },
    }


def build_subject_phase_animation_bundle(
    electrode_ids: list[str],
    traces: dict,
    phase: str,
    tasks: list[str] | None = None,
) -> dict[str, Any]:
    task_list = list(tasks or V1_TASKS)
    return {
        "phase": phase,
        "bundles": {
            key: build_sliding_window_frames(electrode_ids, traces, phase, key, tasks=task_list)
            for key in animation_bundle_keys(task_list)
        },
    }
