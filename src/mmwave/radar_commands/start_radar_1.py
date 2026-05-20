from src.mmwave.mmwavecapture.radar import Radar
from src.mmwave.mmwavecapture import dca1000
import argparse
from pathlib import Path

def main():
    """
    Main function to start the AWR1843BOOST and DCA1000EVM with base configuration.
    """

    parser = argparse.ArgumentParser(description="Start radar 1 with dynamic USB ports.")
    parser.add_argument("-port1", "--config_port", required=True, help="CLI/Configuration port")
    parser.add_argument("-port2", "--data_port", required=True, help="Data port")
    args = parser.parse_args()

    config_port = args.config_port.replace("tty.", "cu.")
    data_port = args.data_port.replace("tty.", "cu.")

    print(f"Starting radar 1 with config_port={config_port} and data_port={data_port}...")

    # Initialize the DCA1000EVM
    dca = dca1000.DCA1000()

    # Initialize the radar
    ROOT_DIR = Path(__file__).resolve().parents[3]
    cfg_file = ROOT_DIR / "src" / "mmwave" / "configs" / "profile_super.cfg" 

    radar = Radar(
        config_port=config_port,
        config_baudrate=115200,
        data_port=data_port,
        data_baudrate=921600,
        config_filename=cfg_file,
        initialize_connection_and_radar=True,
        capture_frames=0,
    )

    # Configure the radar
    radar.config()

    # Check DCA1000EVM connection
    if not dca.system_connection():
        raise RuntimeError(f"DCA1000EVM connection error at {4096}")

    # Initialize DCA1000EVM
    dca.reset_fpga()
    dca.config_fpga()
    dca.config_packet_delay()

    # Start DCA1000EVM
    dca.start_record()
    radar.start_sensor()

    # Get the socket data
    socket_data = dca.get_socket_data("data")
    socket_config = dca.get_socket_data("config")

    # Close the sockets
    socket_data.close()
    socket_config.close()

    print("AWR1843BOOST and DCA1000EVM started successfully.")



if __name__ == "__main__":
    main()