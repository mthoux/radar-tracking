#!/bin/bash

set -e

# --- CONFIGURATION ---
INTERFACE_PATTERN="en6|en7"
PROJECT_DIR="/Users/mathys/Desktop/radar-tracking/src"
CONDA_ENV="radar"
RADAR_1_IP="192.168.33.30"
RADAR_2_IP="192.168.33.32"
NETMASK="255.255.255.0"
SUBNET="192.168.33.0/24"

# 1. Parse arguments
RUN_STREAMING=false
while getopts "r" opt; do
  case $opt in
    r) RUN_STREAMING=true ;;
    *) echo "Usage: $0 [-r]"; exit 1 ;;
  esac
done

# 2. Sudo
sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

cd "$PROJECT_DIR"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Function blindée pour macOS
check_radar_live() {
    local target_ip=$1
    local tmp_file="/tmp/radar_check_${target_ip}.txt"
    echo "Checking traffic for $target_ip..."
    
    # On lance tcpdump en arrière-plan pendant 3 secondes
    # -l : line buffered (important !)
    sudo tcpdump host "$target_ip" -l -c 5 > "$tmp_file" 2>&1 &
    local dump_pid=$!
    
    sleep 3
    
    # On tue le processus s'il tourne encore
    sudo kill $dump_pid > /dev/null 2>&1 || true
    
    # On regarde si le fichier contient des paquets
    if grep -qE "IP|packets captured" "$tmp_file" && grep -q "captured" "$tmp_file" && ! grep -q "0 packets captured" "$tmp_file"; then
        rm -f "$tmp_file"
        return 0
    else
        rm -f "$tmp_file"
        return 1
    fi
}

# 3. Handle Radar 1
if check_radar_live "$RADAR_1_IP"; then
    echo "✅ Radar 1 ($RADAR_1_IP) detected."
    iface_1=$(ifconfig | grep -B 5 "$RADAR_1_IP" | grep -E "en[0-9]" | cut -d: -f1 | head -n 1)
else
    echo "Radar 1 setup required..."
    enX=$(ifconfig -u | grep -E "$INTERFACE_PATTERN" | cut -d: -f1 | head -n 1)
    if [ -z "$enX" ]; then echo "Error: No interface."; exit 1; fi
    sudo ifconfig "$enX" inet "$RADAR_1_IP" netmask "$NETMASK" up
    python -m radar_commands.start_radar_1
    iface_1=$enX
    
    if ! check_radar_live "$RADAR_2_IP"; then
        echo "-------------------------------------------------------"
        echo "ACTION: Swap to Radar 2."
        read -p "Press [Enter] when ready..."
    fi
fi

# 4. Handle Radar 2
if check_radar_live "$RADAR_2_IP"; then
    echo "✅ Radar 2 ($RADAR_2_IP) detected."
else
    enY=$(ifconfig -u | grep -E "$INTERFACE_PATTERN" | grep -v "$iface_1" | cut -d: -f1 | head -n 1)
    if [ -z "$enY" ]; then echo "Error: No interface."; exit 1; fi
    sudo route delete -net "$SUBNET" > /dev/null 2>&1 || true
    sudo ifconfig "$enY" inet "$RADAR_2_IP" netmask "$NETMASK" up
    python -m radar_commands.start_radar_2
fi

# 5. Final
if [ "$RUN_STREAMING" = true ]; then
    python -m streaming.stream
fi