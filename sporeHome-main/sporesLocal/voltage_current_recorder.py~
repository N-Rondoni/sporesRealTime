import time
import csv
import os
import matplotlib.pyplot as plt
import pypalmsens as ps

DATA_FILE = "voltage_current_data.csv"
CALIBRATION_DIR = "calibration_curves"


def ensure_calibration_dir():
    """Creates calibration_curves directory if it doesn't exist."""
    if not os.path.exists(CALIBRATION_DIR):
        os.makedirs(CALIBRATION_DIR)
        print(f"Created directory: {CALIBRATION_DIR}")


def record_current_over_time(volt, duration, sample_interval=0.01, output_file=None):
    """
    Records current over time at a fixed voltage using ChronoAmperometry.
    
    Args:
        volt: Voltage to hold (V)
        duration: Total recording duration (seconds)
        sample_interval: Time between samples (seconds), default 0.01s
        output_file: Output CSV filename (default: calibration_curves/custom_{volt}V.csv)
    
    Returns:
        Tuple of (times, currents) lists
    """
    ensure_calibration_dir()
    
    if output_file is None:
        output_file = os.path.join(CALIBRATION_DIR, f"custom_{volt}V.csv")
    
    print(f'Running ChronoAmperometry: {volt}V for {duration}s, sampling every {sample_interval}s')
    
    method = ps.ChronoAmperometry(
        potential=volt,
        interval_time=sample_interval,
        run_time=duration,
    )
    
    measurement = ps.measure(method)
    df = measurement.dataset.to_dataframe()
    
    # Extract time and current columns
    # Column names may vary - check what's available
    print(f"Available columns: {df.columns.tolist()}")
    
    # Typically 'Time' or 't' for time, 'Current' or 'i' for current
    time_col = [c for c in df.columns if 'time' in c.lower() or c == 't'][0]
    current_col = [c for c in df.columns if 'current' in c.lower() or c == 'i'][0]
    
    times = df[time_col].tolist()
    currents = df[current_col].tolist()
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'current_uA', 'voltage_V'])
        for t, i in zip(times, currents):
            writer.writerow([t, i, volt])
    
    print(f"Recorded {len(times)} samples -> {output_file}")
    return times, currents


def record_measurement(volt, duration, sample_interval=0.1, data_file=DATA_FILE):
    """
    Runs a voltage measurement and appends the steady-state result to CSV file.
    Uses ChronoAmperometry and takes the final (settled) current value.
    """
    times, currents = record_current_over_time(volt, duration, sample_interval)
    
    if currents is None:
        print("Measurement failed, not recording.")
        return None
    
    # Use mean of last 10% of samples as steady-state value
    n_steady = max(1, len(currents) // 10)
    steady_state_current = sum(currents[-n_steady:]) / n_steady
    
    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(data_file)
    
    with open(data_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['voltage_V', 'current_uA', 'timestamp'])
        writer.writerow([volt, steady_state_current, time.strftime('%Y-%m-%d %H:%M:%S')])
    
    print(f"Recorded steady-state: {volt} V, {steady_state_current:.4f} µA -> {data_file}")
    return steady_state_current


def plot_vi_curve(data_file=DATA_FILE):
    """
    Plots voltage vs current from the recorded data file.
    """
    if not os.path.isfile(data_file):
        print(f"No data file found: {data_file}")
        return
    
    voltages = []
    currents = []
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            voltages.append(float(row['voltage_V']))
            currents.append(float(row['current_uA']))
    
    plt.figure(figsize=(8, 6))
    plt.plot(voltages, currents, 'bo-', markersize=8, linewidth=1.5)
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Current (µA)', fontsize=12)
    plt.title('Voltage vs Current', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vi_curve.png', dpi=150)
    plt.show()
    print("Plot saved to vi_curve.png")


def clear_data(data_file=DATA_FILE):
    """
    Clears the data file to start fresh.
    """
    if os.path.isfile(data_file):
        os.remove(data_file)
        print(f"Cleared {data_file}")


def plot_current_over_time(data_file=None, volt=None, show_points=True, smooth_window=None):
    """
    Plots current vs time from a time-series recording.
    
    Args:
        data_file: CSV file to plot. If None, uses default naming with volt.
        volt: Voltage used (for default filename and title)
        show_points: If True, show individual sample points (default True)
        smooth_window: If set, apply moving average with this window size
    """
    if data_file is None:
        if volt is None:
            print("Provide either data_file or volt")
            return
        data_file = os.path.join(CALIBRATION_DIR, f"custom_{volt}V.csv")
    
    if not os.path.isfile(data_file):
        print(f"No data file found: {data_file}")
        return
    
    times = []
    currents = []
    voltage = None
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            currents.append(float(row['current_uA']))
            if voltage is None:
                voltage = float(row['voltage_V'])
    
    # Optional smoothing (moving average)
    if smooth_window and smooth_window > 1:
        smoothed = []
        for i in range(len(currents)):
            start = max(0, i - smooth_window // 2)
            end = min(len(currents), i + smooth_window // 2 + 1)
            smoothed.append(sum(currents[start:end]) / (end - start))
        currents_plot = smoothed
        smooth_label = f' (smoothed, window={smooth_window})'
    else:
        currents_plot = currents
        smooth_label = ''
    
    plt.figure(figsize=(10, 6))
    
    if show_points and not smooth_window:
        plt.plot(times, currents_plot, 'b.', markersize=4, label='Samples')
        plt.plot(times, currents_plot, 'b-', linewidth=0.5, alpha=0.3)
    else:
        plt.plot(times, currents_plot, 'b-', linewidth=0.8)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Current (µA)', fontsize=12)
    plt.title(f'Current vs Time at {voltage}V ({len(times)} samples){smooth_label}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add stats annotation (from raw data)
    avg_current = sum(currents) / len(currents)
    min_current = min(currents)
    max_current = max(currents)
    stats_text = f'Mean: {avg_current:.4f} µA\nMin: {min_current:.4f} µA\nMax: {max_current:.4f} µA'
    plt.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_file = data_file.replace('.csv', '.png')
    plt.savefig(plot_file, dpi=150)
    #plt.show()
    print(f"Plot saved to {plot_file}")


def build_vi_curve_from_timeseries(voltage_list, skip_seconds=6.0, output_file=None):
    """
    Steps through time-series CSVs for each voltage and extracts steady-state
    currents to build a V-I curve.
    
    Args:
        voltage_list: List of voltages that have been recorded
        skip_seconds: Seconds to skip at start to avoid transients (default 6.0)
        output_file: Output CSV filename
    
    Returns:
        Tuple of (voltages, currents, std_devs) lists
    """
    ensure_calibration_dir()
    
    if output_file is None:
        output_file = os.path.join(CALIBRATION_DIR, "vi_steady_state.csv")
    
    voltages = []
    currents = []
    std_devs = []
    
    for volt in voltage_list:
        data_file = os.path.join(CALIBRATION_DIR, f"custom_{volt}V.csv")
        if not os.path.isfile(data_file):
            print(f"Warning: {data_file} not found, skipping")
            continue
        
        v, i_avg, std = extract_steady_state(data_file, skip_seconds)
        if i_avg is not None:
            voltages.append(v)
            currents.append(i_avg)
            std_devs.append(std)
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['voltage_V', 'current_uA', 'std_uA'])
        for v, i, s in zip(voltages, currents, std_devs):
            writer.writerow([v, i, s])
    
    print(f"\nSaved {len(voltages)} V-I pairs -> {output_file}")
    return voltages, currents, std_devs


def extract_steady_state(data_file, skip_seconds=6.0):
    """
    Extracts the steady-state current from a time-series CSV by averaging
    values after skip_seconds to avoid transients.
    
    Args:
        data_file: CSV file from record_current_over_time
        skip_seconds: Seconds to skip at the start (default 6.0)
    
    Returns:
        Tuple of (voltage, steady_state_current, std_dev)
    """
    times = []
    currents = []
    voltage = None
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            currents.append(float(row['current_uA']))
            if voltage is None:
                voltage = float(row['voltage_V'])
    
    # Filter to samples after skip_seconds
    steady_currents = [c for t, c in zip(times, currents) if t >= skip_seconds]
    
    if not steady_currents:
        print(f"Warning: No samples after {skip_seconds}s in {data_file}")
        return voltage, None, None
    
    avg = sum(steady_currents) / len(steady_currents)
    variance = sum((c - avg) ** 2 for c in steady_currents) / len(steady_currents)
    std = variance ** 0.5
    
    print(f"{data_file}: V={voltage}V, I_steady={avg:.6f} µA (std={std:.6f}, n={len(steady_currents)})")
    return voltage, avg, std


def parse_pstrace_csv(filepath, skip_seconds=6.0):
    """
    Parses a PSTrace CSV file (UTF-16 encoded) and extracts steady-state current.
    
    Args:
        filepath: Path to PSTrace CSV file
        skip_seconds: Seconds to skip at start to avoid transients
    
    Returns:
        Tuple of (times, currents, steady_state_avg, steady_state_std)
    """
    times = []
    currents = []
    
    # Read with UTF-16 encoding
    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()
    
    # Find the data start (after header line "s,µA" or similar)
    data_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip header lines until we hit numeric data
        if not data_started:
            # Check if line starts with a number (data row)
            try:
                parts = line.split(',')
                float(parts[0])
                data_started = True
            except (ValueError, IndexError):
                continue
        
        if data_started:
            try:
                parts = line.split(',')
                t = float(parts[0])
                i = float(parts[1])
                times.append(t)
                currents.append(i)
            except (ValueError, IndexError):
                continue
    
    # Calculate steady-state (after skip_seconds)
    steady_currents = [c for t, c in zip(times, currents) if t >= skip_seconds]
    
    if steady_currents:
        avg = sum(steady_currents) / len(steady_currents)
        variance = sum((c - avg) ** 2 for c in steady_currents) / len(steady_currents)
        std = variance ** 0.5
    else:
        avg, std = None, None
    
    print(f"PSTrace file: {len(times)} samples, t=[{min(times):.1f}, {max(times):.1f}]s")
    print(f"Steady-state (t>{skip_seconds}s): I={avg:.6f} µA (std={std:.6f}, n={len(steady_currents)})")
    
    return times, currents, avg, std


def compare_implementations(custom_file, pstrace_file, skip_seconds=6.0, output_file=None):
    """
    Compares steady-state values between custom implementation and PSTrace.
    
    Args:
        custom_file: Path to custom implementation CSV (from record_current_over_time)
        pstrace_file: Path to PSTrace CSV file
        skip_seconds: Seconds to skip for steady-state calculation
        output_file: Output CSV for comparison results
    
    Returns:
        Dict with comparison results
    """
    ensure_calibration_dir()
    
    if output_file is None:
        output_file = os.path.join(CALIBRATION_DIR, "calibration_comparison.csv")
    
    # Parse custom file
    custom_times = []
    custom_currents = []
    voltage = None
    
    with open(custom_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            custom_times.append(float(row['time_s']))
            custom_currents.append(float(row['current_uA']))
            if voltage is None:
                voltage = float(row['voltage_V'])
    
    custom_steady = [c for t, c in zip(custom_times, custom_currents) if t >= skip_seconds]
    custom_avg = sum(custom_steady) / len(custom_steady)
    custom_std = (sum((c - custom_avg) ** 2 for c in custom_steady) / len(custom_steady)) ** 0.5
    
    # Parse PSTrace file
    ps_times, ps_currents, ps_avg, ps_std = parse_pstrace_csv(pstrace_file, skip_seconds)
    
    # Calculate difference
    diff = custom_avg - ps_avg
    diff_percent = (diff / ps_avg * 100) if ps_avg != 0 else 0
    
    results = {
        'voltage_V': voltage,
        'custom_avg_uA': custom_avg,
        'custom_std_uA': custom_std,
        'pstrace_avg_uA': ps_avg,
        'pstrace_std_uA': ps_std,
        'diff_uA': diff,
        'diff_percent': diff_percent
    }
    
    # Save comparison
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results.keys())
        writer.writerow(results.values())
    
    print(f"\n=== Comparison at {voltage}V ===")
    print(f"Custom:  {custom_avg:.6f} ± {custom_std:.6f} µA")
    print(f"PSTrace: {ps_avg:.6f} ± {ps_std:.6f} µA")
    print(f"Diff:    {diff:.6f} µA ({diff_percent:.2f}%)")
    print(f"Saved to {output_file}")
    
    return results


def plot_comparison(voltages=None, skip_seconds=6.0):
    """
    Plots V-I curves comparing custom implementation vs PSTrace.
    Loads files from calibration_curves directory.
    
    Expected file naming:
        - Custom: calibration_curves/custom_{volt}V.csv  (e.g., custom_0.5V.csv)
        - PSTrace: calibration_curves/PStrace_range_{volt}.csv  (e.g., PStrace_range_0_5.csv)
    
    Args:
        voltages: List of voltage values to compare. If None, auto-detects from files.
        skip_seconds: Seconds to skip for steady-state calculation
    """
    # Auto-detect voltages if not provided
    if voltages is None:
        voltages = []
        for f in os.listdir(CALIBRATION_DIR):
            if f.startswith("custom_") and f.endswith("V.csv"):
                volt_str = f.replace("custom_", "").replace("V.csv", "")
                try:
                    voltages.append(float(volt_str))
                except ValueError:
                    continue
        voltages = sorted(voltages)
        print(f"Auto-detected voltages: {voltages}")
    
    custom_currents = []
    pstrace_currents = []
    valid_voltages = []
    
    for volt in voltages:
        custom_file = os.path.join(CALIBRATION_DIR, f"custom_{volt}V.csv")
        # PSTrace uses underscore for decimal: 0.5 -> 0_5
        volt_str = str(volt).replace(".", "_")
        pstrace_file = os.path.join(CALIBRATION_DIR, f"PStrace_range_{volt_str}.csv")
        
        if not os.path.isfile(custom_file):
            print(f"Warning: {custom_file} not found, skipping {volt}V")
            continue
        if not os.path.isfile(pstrace_file):
            print(f"Warning: {pstrace_file} not found, skipping {volt}V")
            continue
        
        # Get custom steady-state
        _, custom_avg, _ = extract_steady_state(custom_file, skip_seconds)
        
        # Get PSTrace steady-state
        _, _, ps_avg, _ = parse_pstrace_csv(pstrace_file, skip_seconds)
        
        if custom_avg is not None and ps_avg is not None:
            valid_voltages.append(volt)
            custom_currents.append(custom_avg)
            pstrace_currents.append(ps_avg)
    
    if not valid_voltages:
        print("No valid voltage pairs found. Check that files exist in calibration_curves/")
        return
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    plt.plot(valid_voltages, pstrace_currents, 'bo-', linewidth=1.5, markersize=8, label='PSTrace')
    plt.plot(valid_voltages, custom_currents, 'rs-', linewidth=1.5, markersize=8, label='Custom')
    
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Current (µA)', fontsize=12)
    plt.title('V-I Curve: Custom vs PSTrace', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_plot = os.path.join(CALIBRATION_DIR, "vi_comparison.png")
    plt.savefig(output_plot, dpi=150)
    plt.show()
    print(f"Plot saved to {output_plot}")


def run_voltage_sweep(voltage_list, duration=30, sample_interval=0.01, skip_seconds=6.0):
    """
    Runs a full voltage sweep: records time-series at each voltage,
    then extracts steady-state values into a V-I curve.
    
    Args:
        voltage_list: List of voltages to test
        duration: Recording duration at each voltage (seconds)
        sample_interval: Time between samples (seconds)
        skip_seconds: Seconds to skip when computing steady-state
    
    Returns:
        Tuple of (voltages, currents, std_devs) lists
    """
    print(f"Starting voltage sweep: {voltage_list}")
    print(f"Duration per voltage: {duration}s, skip first {skip_seconds}s for steady-state\n")
    
    for volt in voltage_list:
        record_current_over_time(volt, duration, sample_interval)
        print()
    
    return build_vi_curve_from_timeseries(voltage_list, skip_seconds)

def run_voltage_pattern_good(pattern, sample_interval=0.1, output_file=None, num_cycles=None, total_duration=None):
    """
    Runs a voltage pattern using ChronoAmperometry for accurate timing.
    
    Args:
        pattern: List of (voltage, duration_seconds) tuples
                 e.g., [(0.5, 300), (-2.0, 30)] for +0.5V 5min, -2V 30s
        sample_interval: Time between samples (seconds)
        output_file: Output CSV filename (default: voltage_pattern_{timestamp}.csv)
        num_cycles: Number of times to repeat the pattern (optional)
        total_duration: Total experiment duration in seconds (optional)
                        Pattern repeats until this time is reached
    
    Returns:
        Tuple of (times, voltages, currents) lists
    """
    ensure_calibration_dir()
    
    if output_file is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(CALIBRATION_DIR, f"voltage_pattern_{timestamp}.csv")
    
    # Calculate cycle duration
    cycle_duration = sum(duration for _, duration in pattern)
    
    # Determine number of cycles
    if num_cycles is not None:
        cycles = num_cycles
    elif total_duration is not None:
        cycles = int(total_duration // cycle_duration)
        if cycles == 0:
            cycles = 1
        print(f"Pattern cycle: {cycle_duration}s, total duration: {total_duration}s -> {cycles} complete cycles")
    else:
        cycles = 1
    
    total_expected = cycles * cycle_duration
    print(f"Running {cycles} cycle(s), total time: {total_expected}s ({total_expected/60:.1f} min)")
    print(f"Pattern: {pattern}")
    print()
    
    all_times = []
    all_voltages = []
    all_currents = []
    global_time_offset = 0
    
    for cycle in range(cycles):
        print(f"=== Cycle {cycle + 1}/{cycles} ===")
        
        for volt, duration in pattern:
            print(f"  Applying {volt}V for {duration}s ({duration/60:.1f} min)...")
            
            method = ps.ChronoAmperometry(
                potential=volt,
                interval_time=sample_interval,
                run_time=duration,
            )
            
            measurement = ps.measure(method)
            df = measurement.dataset.to_dataframe()
            
            # Extract columns
            time_col = [c for c in df.columns if 'time' in c.lower() or c == 't'][0]
            current_col = [c for c in df.columns if 'current' in c.lower() or c == 'i'][0]
            
            times = df[time_col].tolist()
            currents = df[current_col].tolist()
            
            # Add to global lists with time offset
            for t, i in zip(times, currents):
                all_times.append(t + global_time_offset)
                all_voltages.append(volt)
                all_currents.append(i)
            
            global_time_offset += duration
            print(f"    Completed. Total elapsed: {global_time_offset}s ({global_time_offset/60:.1f} min)")
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'voltage_V', 'current_uA'])
        for t, v, i in zip(all_times, all_voltages, all_currents):
            writer.writerow([t, v, i])
    
    print(f"\nExperiment complete. Recorded {len(all_times)} samples -> {output_file}")
    return all_times, all_voltages, all_currents

def run_voltage_pattern(pattern, sample_interval=0.1, output_file=None, num_cycles=None, total_duration=None, verbose=True):
    """
    Runs a voltage pattern using ChronoAmperometry for accurate timing.
    
    Args:
        pattern: List of (voltage, duration_seconds) tuples
                 e.g., [(0.5, 300), (-2.0, 30)] for +0.5V 5min, -2V 30s
        sample_interval: Time between samples (seconds)
        output_file: Output CSV filename (default: voltage_pattern_{timestamp}.csv)
        num_cycles: Number of times to repeat the pattern (optional)
        total_duration: Total experiment duration in seconds (optional)
                        Pattern repeats until this time is reached
        verbose: If True, print real-time progress during measurement
    
    Returns:
        Tuple of (times, voltages, currents) lists
    """
    ensure_calibration_dir()
    
    if output_file is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(CALIBRATION_DIR, f"voltage_pattern_{timestamp}.csv")
    
    # Calculate cycle duration
    cycle_duration = sum(duration for _, duration in pattern)
    
    # Determine number of cycles
    if num_cycles is not None:
        cycles = num_cycles
    elif total_duration is not None:
        cycles = int(total_duration // cycle_duration)
        if cycles == 0:
            cycles = 1
        print(f"Pattern cycle: {cycle_duration}s, total duration: {total_duration}s -> {cycles} complete cycles")
    else:
        cycles = 1
    
    total_expected = cycles * cycle_duration
    print(f"Running {cycles} cycle(s), total time: {total_expected}s ({total_expected/60:.1f} min)")
    print(f"Pattern: {pattern}")
    print()
    
    all_times = []
    all_voltages = []
    all_currents = []
    global_time_offset = 0
    
    # Progress callback for real-time feedback
    last_print_time = [0]  # Use list to allow modification in nested function
    
    def progress_callback(data):
        if not verbose:
            return
        try:
            # Use correct attribute names: x_array, y_array
            if data.x_array and len(data.x_array) > 0:
                current_time = data.x_array[-1]
                if current_time - last_print_time[0] >= 10:
                    last_print_time[0] = current_time
                    latest_current = data.y_array[-1] if data.y_array else 0
                    elapsed_total = global_time_offset + current_time
                    print(f"    t={elapsed_total:.0f}s ({elapsed_total/60:.1f}min) | I={latest_current:.4f} µA")
        except Exception as e:
            pass  # Silently ignore callback errors
    
    for cycle in range(cycles):
        print(f"=== Cycle {cycle + 1}/{cycles} ===")
        
        for volt, duration in pattern:
            print(f"  Applying {volt}V for {duration}s ({duration/60:.1f} min)...")
            last_print_time[0] = 0  # Reset for each step
            
            method = ps.ChronoAmperometry(
                potential=volt,
                interval_time=sample_interval,
                run_time=duration,
            )
            
            measurement = ps.measure(method, callback=progress_callback if verbose else None)
            df = measurement.dataset.to_dataframe()
            
            # Extract columns
            time_col = [c for c in df.columns if 'time' in c.lower() or c == 't'][0]
            current_col = [c for c in df.columns if 'current' in c.lower() or c == 'i'][0]
            
            times = df[time_col].tolist()
            currents = df[current_col].tolist()
            
            # Add to global lists with time offset
            for t, i in zip(times, currents):
                all_times.append(t + global_time_offset)
                all_voltages.append(volt)
                all_currents.append(i)
            
            global_time_offset += duration
            print(f"    Completed. Total elapsed: {global_time_offset}s ({global_time_offset/60:.1f} min)")
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'voltage_V', 'current_uA'])
        for t, v, i in zip(all_times, all_voltages, all_currents):
            writer.writerow([t, v, i])
    
    print(f"\nExperiment complete. Recorded {len(all_times)} samples -> {output_file}")
    return all_times, all_voltages, all_currents


def plot_voltage_pattern(data_file=None, show_voltage=True):
    """
    Plots results from a voltage pattern experiment.
    
    Args:
        data_file: CSV file from run_voltage_pattern. If None, uses most recent.
        show_voltage: If True, show voltage trace on secondary y-axis
    """
    if data_file is None:
        # Find most recent pattern file
        pattern_files = [f for f in os.listdir(CALIBRATION_DIR) if f.startswith("voltage_pattern_")]
        if not pattern_files:
            print("No voltage pattern files found")
            return
        data_file = os.path.join(CALIBRATION_DIR, sorted(pattern_files)[-1])
        print(f"Using: {data_file}")
    
    times = []
    voltages = []
    currents = []
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            voltages.append(float(row['voltage_V']))
            currents.append(float(row['current_uA']))
    
    # Convert to minutes for readability
    times_min = [t / 60 for t in times]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Current trace
    ax1.plot(times_min, currents, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('Current (µA)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    if show_voltage:
        # Voltage trace on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(times_min, voltages, 'r-', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Voltage (V)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Voltage Pattern Experiment', fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = data_file.replace('.csv', '.png')
    plt.savefig(plot_file, dpi=150)
    plt.show()
    print(f"Plot saved to {plot_file}")

def run_voltage_pattern_bad(pattern, sample_interval=0.1, output_file=None, num_cycles=None, total_duration=None, verbose=True):
    """
    Runs a voltage pattern using ChronoAmperometry for accurate timing.
    
    Args:
        pattern: List of (voltage, duration_seconds) tuples
                 e.g., [(0.5, 300), (-2.0, 30)] for +0.5V 5min, -2V 30s
        sample_interval: Time between samples (seconds)
        output_file: Output CSV filename (default: voltage_pattern_{timestamp}.csv)
        num_cycles: Number of times to repeat the pattern (optional)
        total_duration: Total experiment duration in seconds (optional)
                        Pattern repeats until this time is reached
        verbose: If True, print real-time progress during measurement
    
    Returns:
        Tuple of (times, voltages, currents) lists
    """
    ensure_calibration_dir()
    
    if output_file is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(CALIBRATION_DIR, f"voltage_pattern_{timestamp}.csv")
    
    # Calculate cycle duration
    cycle_duration = sum(duration for _, duration in pattern)
    
    # Determine number of cycles
    if num_cycles is not None:
        cycles = num_cycles
    elif total_duration is not None:
        cycles = int(total_duration // cycle_duration)
        if cycles == 0:
            cycles = 1
        print(f"Pattern cycle: {cycle_duration}s, total duration: {total_duration}s -> {cycles} complete cycles")
    else:
        cycles = 1
    
    total_expected = cycles * cycle_duration
    print(f"Running {cycles} cycle(s), total time: {total_expected}s ({total_expected/60:.1f} min)")
    print(f"Pattern: {pattern}")
    print()
    
    all_times = []
    all_voltages = []
    all_currents = []
    global_time_offset = 0
    
    # Progress callback for real-time feedback
    last_print_time = [0]  # Use list to allow modification in nested function
    
    def progress_callback(data):
        if not verbose:
            return
        # Print every 10 seconds
        if data.x and len(data.x) > 0:
            current_time = data.x[-1]
            if current_time - last_print_time[0] >= 10:
                last_print_time[0] = current_time
                latest_current = data.y[-1] if data.y else 0
                elapsed_total = global_time_offset + current_time
                print(f"    t={elapsed_total:.0f}s ({elapsed_total/60:.1f}min) | I={latest_current:.4f} µA")
    
    for cycle in range(cycles):
        print(f"=== Cycle {cycle + 1}/{cycles} ===")
        
        for volt, duration in pattern:
            print(f"  Applying {volt}V for {duration}s ({duration/60:.1f} min)...")
            last_print_time[0] = 0  # Reset for each step
            
            method = ps.ChronoAmperometry(
                potential=volt,
                interval_time=sample_interval,
                run_time=duration,
            )
            
            measurement = ps.measure(method, callback=progress_callback if verbose else None)
            df = measurement.dataset.to_dataframe()
            
            # Extract columns
            time_col = [c for c in df.columns if 'time' in c.lower() or c == 't'][0]
            current_col = [c for c in df.columns if 'current' in c.lower() or c == 'i'][0]
            
            times = df[time_col].tolist()
            currents = df[current_col].tolist()
            
            # Add to global lists with time offset
            for t, i in zip(times, currents):
                all_times.append(t + global_time_offset)
                all_voltages.append(volt)
                all_currents.append(i)
            
            global_time_offset += duration
            print(f"    Completed. Total elapsed: {global_time_offset}s ({global_time_offset/60:.1f} min)")
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'voltage_V', 'current_uA'])
        for t, v, i in zip(all_times, all_voltages, all_currents):
            writer.writerow([t, v, i])
    
    print(f"\nExperiment complete. Recorded {len(all_times)} samples -> {output_file}")
    return all_times, all_voltages, all_currents


# Example usage
if __name__ == "__main__":
    # Record current over time at fixed voltage
    #record_current_over_time(-0.8, duration=300, sample_interval=0.1)
    
    #plot_current_over_time(volt=-0.8)

    #plot_current_over_time(volt=0.5, smooth_window=20)
    
    #record_current_over_time(1.0, duration=30, sample_interval=0.01)
    
    #plot_current_over_time(volt=1.0)

    #plot_current_over_time(volt=1.0, smooth_window=40)

    # Compare custom vs PSTrace (expects files in calibration_curves/)
    # plot_comparison(voltages=[0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Or auto-detect voltages
    #plot_comparison()

    # Define pattern: (voltage, duration_seconds)
    pattern = [
    (-0.5, 300),    # -0.5V for 5 min
    (0.5, 30),      # 0.5V for 30s reset
    ]
    
    #pattern = [
    #(0.5, 30),    # +0.5V for 5 min
    #(-0.5, 10),    # -2V for 30s reset
    #]
   
    # Option 1: Run for specific number of cycles
    run_voltage_pattern(pattern, sample_interval=0.1, num_cycles=5, verbose=True)

    # Option 2: Run for total duration (fits as many complete cycles as possible)
    #run_voltage_pattern(pattern, sample_interval=0.1, total_duration=30*60)
    plot_voltage_pattern()

    plt.show()
