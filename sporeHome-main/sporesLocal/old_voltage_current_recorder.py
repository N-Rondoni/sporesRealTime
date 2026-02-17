import time
import csv
import os
import matplotlib.pyplot as plt
import pypalmsens as ps

DATA_FILE = "voltage_current_data.csv"


def record_current_over_time(volt, duration, sample_interval=0.01, output_file=None):
    """
    Records current over time at a fixed voltage using ChronoAmperometry.
    
    Args:
        volt: Voltage to hold (V)
        duration: Total recording duration (seconds)
        sample_interval: Time between samples (seconds), default 0.01s
        output_file: Output CSV filename (default: current_vs_time_{volt}V.csv)
    
    Returns:
        Tuple of (times, currents) lists
    """
    if output_file is None:
        output_file = f"current_vs_time_{volt}V.csv"
    
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
        data_file = f"current_vs_time_{volt}V.csv"
    
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

def build_vi_curve_from_timeseries(voltage_list, skip_seconds=6.0, output_file="vi_steady_state.csv"):
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
    voltages = []
    currents = []
    std_devs = []
    
    for volt in voltage_list:
        data_file = f"current_vs_time_{volt}V.csv"
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


def compare_implementations(custom_file, pstrace_file, skip_seconds=6.0, output_file="calibration_comparison.csv"):
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


def plot_comparison(custom_files, pstrace_files, voltages, skip_seconds=6.0):
    """
    Plots V-I curves comparing custom implementation vs PSTrace.
    
    Args:
        custom_files: List of custom CSV file paths
        pstrace_files: List of PSTrace CSV file paths
        voltages: List of voltage values corresponding to each file pair
        skip_seconds: Seconds to skip for steady-state calculation
    """
    custom_currents = []
    pstrace_currents = []
    
    for custom_file, pstrace_file, volt in zip(custom_files, pstrace_files, voltages):
        # Get custom steady-state
        _, custom_avg, _ = extract_steady_state(custom_file, skip_seconds)
        custom_currents.append(custom_avg)
        
        # Get PSTrace steady-state
        _, _, ps_avg, _ = parse_pstrace_csv(pstrace_file, skip_seconds)
        pstrace_currents.append(ps_avg)
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    plt.plot(voltages, pstrace_currents, 'bo-', linewidth=1.5, markersize=8, label='PSTrace')
    plt.plot(voltages, custom_currents, 'rs-', linewidth=1.5, markersize=8, label='Custom')
    
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Current (µA)', fontsize=12)
    plt.title('V-I Curve: Custom vs PSTrace', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('vi_comparison.png', dpi=150)
    plt.show()
    print("Plot saved to vi_comparison.png")

# Example usage
if __name__ == "__main__":
    # Record current over time at fixed voltage
    #record_current_over_time(0.5, duration=30, sample_interval=0.01)
    
    #plot_current_over_time(volt=0.5)

    #plot_current_over_time(volt=0.5, smooth_window=20)
    
    record_current_over_time(1.0, duration=30, sample_interval=0.01)
    
    plot_current_over_time(volt=1.0)

    plot_current_over_time(volt=1.0, smooth_window=40)


    plt.show()
    # Or build a V-I curve
    # clear_data()
    # for v in [0.0, 0.5, 1.0, 1.5, 2.0]:
    #     record_measurement(v, duration=10)
    # plot_vi_curve()
