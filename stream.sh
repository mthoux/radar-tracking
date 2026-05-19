#!/bin/bash

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
# Network interfaces to probe/configure
INTF_PRIMARY="en6"
INTF_SECONDARY="en7"

# IP addresses and Subnets
SUBNET_TARGET="192.168.33.0/24"         # Used for route cleanup
RADAR_TARGET_IP="192.168.33.180"        # IP to scan for with arp-scan

HOST_IP_RADAR_1="192.168.33.30"         # IP assigned to your Mac for Radar 1 connection
HOST_IP_RADAR_2="192.168.33.32"         # IP assigned to your Mac for Radar 2 connection
NETMASK="255.255.255.0"                 # Network mask

# Conda environment name
CONDA_ENV_NAME="radar"

# ==========================================
# PATH INDEPENDENCY SETUP
# ==========================================
# 1. Start with the directory where launch.sh lives
CURRENT_LOOKUP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Walk backwards up the tree until we find the real project root (containing 'src')
while [ "$CURRENT_LOOKUP" != "/" ]; do
    if [ -d "$CURRENT_LOOKUP/src" ]; then
        PROJECT_ROOT="$CURRENT_LOOKUP"
        break
    fi
    CURRENT_LOOKUP="$( cd "$CURRENT_LOOKUP/.." && pwd )"
done

# 3. Fail-safe fallback if we somehow didn't find 'src'
if [ -z "$PROJECT_ROOT" ]; then
    echo "🚨 Error: Could not locate the project root containing the 'src' directory."
    exit 1
fi

# Inject PROJECT_ROOT into Python's module search paths
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Move inside the project root for execution safety
cd "$PROJECT_ROOT" || exit 1

# ==========================================
# CONDA ENVIRONMENT CHECK
# ==========================================
echo "🐍 Checking Conda environment..."

if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    echo "🔄 '$CONDA_ENV_NAME' environment not active. Attempting activation..."
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    
    if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
        echo "🚨 Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."
        exit 1
    fi
fi

echo "✅ Conda environment '$CONDA_ENV_NAME' is active."
echo "----------------------------------"

# ==========================================
# RADAR INITIALIZATION BLOCK (-init)
# ==========================================
# Check if the user passed the "-init" argument
if [ "$1" == "-init" ]; then
    echo "⚡ [-init flag detected] Starting hardware setup..."
    echo "----------------------------------"

    # ==========================================
    # RADAR DETECTION & NETWORK SETUP
    # ==========================================
    echo "🔍 Scanning for connected radars..."

    # Clean up routing tables using config variable
    sudo route delete -net "$SUBNET_TARGET" 2>/dev/null

    # 1. Search for target IP on primary interface
    scan_primary=$(sudo arp-scan --interface="$INTF_PRIMARY" "$RADAR_TARGET_IP" 2>/dev/null)

    # 2. Map network interfaces dynamically based on search result
    if echo "$scan_primary" | grep -q "$RADAR_TARGET_IP"; then
        EN_RADAR_1="$INTF_PRIMARY"
        EN_RADAR_2="$INTF_SECONDARY"
    else
        EN_RADAR_1="$INTF_SECONDARY"
        EN_RADAR_2="$INTF_PRIMARY"
    fi

    echo "✅ EN_RADAR_1=$EN_RADAR_1"
    echo "✅ EN_RADAR_2=$EN_RADAR_2"
    echo "----------------------------------"

    # ==========================================
    # DYNAMIC USB PORT DETECTION
    # ==========================================
    echo "🔌 Detecting available USB ports..."

    # Pair A: Purely numerical ports sorted
    PAIRE_A=($(ls /dev/cu.usbmodem* 2>/dev/null | grep -v "usbmodemR" | sort))
    PAIRE_A_P1="${PAIRE_A[0]}"
    PAIRE_A_P2="${PAIRE_A[1]}"

    # Pair B: Static serial number ports
    PAIRE_B=($(ls /dev/cu.usbmodemR* 2>/dev/null | sort))
    PAIRE_B_P1="${PAIRE_B[0]}"
    PAIRE_B_P2="${PAIRE_B[1]}"

    # Fail-safe check
    if [ -z "$PAIRE_A_P1" ] || [ -z "$PAIRE_B_P1" ]; then
        echo "🚨 Error: Could not detect all 4 required USB ports."
        exit 1
    fi

    echo "📍 Found PAIR A (Dynamic): $PAIRE_A_P1 & $PAIRE_A_P2"
    echo "📍 Found PAIR B (Serial):  $PAIRE_B_P1 & $PAIRE_B_P2"
    echo "----------------------------------"

    # ==========================================
    # CONFIGURE & START RADAR 1
    # ==========================================
    echo "⚙️ Setting up network for Radar 1 ($HOST_IP_RADAR_1)..."
    sudo ifconfig "$EN_RADAR_1" inet "$HOST_IP_RADAR_1" netmask "$NETMASK" up

    # --- TENTATIVE 1: PAIR B FIRST (Fixed Serial Ports) ---
    echo "🚀 [Radar 1] Attempt 1: PAIR B - Fixed Serial ($PAIRE_B_P1, $PAIRE_B_P2)..."
    python -m src.mmwave.radar_commands.start_radar_1 -port1 "$PAIRE_B_P1" -port2 "$PAIRE_B_P2"

    if [ $? -ne 0 ]; then
        echo "⚠️ [Radar 1] Pair B failed. Trying PAIR A - Normal ($PAIRE_A_P1, $PAIRE_A_P2)..."
        
        # --- TENTATIVE 2: FALLBACK TO PAIR A (Normal Order) ---
        python -m src.mmwave.radar_commands.start_radar_1 -port1 "$PAIRE_A_P1" -port2 "$PAIRE_A_P2"
        
        if [ $? -ne 0 ]; then
            echo "⚠️ [Radar 1] Pair A Normal failed. Trying PAIR A - INVERTED ($PAIRE_A_P2, $PAIRE_A_P1)..."
            
            # --- TENTATIVE 3: FALLBACK TO PAIR A (Inverted Order) ---
            python -m src.mmwave.radar_commands.start_radar_1 -port1 "$PAIRE_A_P2" -port2 "$PAIRE_A_P1"
            
            if [ $? -ne 0 ]; then
                echo "----------------------------------------------------------------------"
                echo "🚨 [Radar 1] Critical failure on all port combinations. Halting."
                echo "💡 TIP: If you launched too early after power up, try to launch again."
                echo "   It may be that the radar connections were not fully set up yet."
                echo "   Also make sure to execute this script in terminal not is VS code"
                echo "----------------------------------------------------------------------"
                exit 1
            fi
            
            echo "✅ [Radar 1] Started successfully on PAIR A (Inverted)!"
            R2_P1="$PAIRE_B_P1"
            R2_P2="$PAIRE_B_P2"
        else
            echo "✅ [Radar 1] Started successfully on PAIR A (Normal)!"
            R2_P1="$PAIRE_B_P1"
            R2_P2="$PAIRE_B_P2"
        fi
    else
        echo "✅ [Radar 1] Started successfully on PAIR B (Fixed)!"
        R2_P1="$PAIRE_A_P1"
        R2_P2="$PAIRE_A_P2"
    fi

    # Clean up routing tables before configuring Radar 2
    sudo route delete -net "$SUBNET_TARGET" 2>/dev/null

    # ==========================================
    # CONFIGURE & START RADAR 2
    # ==========================================
    echo "⚙️ Setting up network for Radar 2 ($HOST_IP_RADAR_2)..."
    sudo ifconfig "$EN_RADAR_2" inet "$HOST_IP_RADAR_2" netmask "$NETMASK" up

    echo "🚀 [Radar 2] Automatically starting with remaining ports ($R2_P1, $R2_P2)..."
    python -m src.mmwave.radar_commands.start_radar_2 -port1 "$R2_P1" -port2 "$R2_P2"

    # Catch Pair A inversion fallback for Radar 2 if it's the one using it
    if [ $? -ne 0 ] && [ "$R2_P1" == "$PAIRE_A_P1" ]; then
        echo "⚠️ [Radar 2] Failed with Pair A Normal. Trying Pair A Inverted..."
        python -m src.mmwave.radar_commands.start_radar_2 -port1 "$PAIRE_A_P2" -port2 "$PAIRE_A_P1"
    fi

    if [ $? -eq 0 ]; then
        echo "✅ [Radar 2] Started successfully!"
    else
        echo "🚨 [Radar 2] Script execution failed."
        exit 1
    fi
    echo "----------------------------------"
fi

# ==========================================
# PYTHON STREAM PROCESSING (ALWAYS RUNS)
# ==========================================
echo "🖥️ Starting processing stream..."
python -m src.processing.stream