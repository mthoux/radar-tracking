from mmwavecapture import dca1000

def stop_dca_only(config_dca):
    # On initialise uniquement le DCA
    dca = dca1000.DCA1000(config=config_dca)
    
    ip_dca = config_dca['ethernetConfig']['DCA1000IPAddress']
    print(f"Tentative d'arrêt du flux DCA1000 à l'adresse {ip_dca}...")
    
    try:
        # Cette commande dit à la carte DCA d'arrêter d'envoyer des paquets UDP
        success = dca.stop_record()
        if success:
            print(f"✅ Flux arrêté avec succès pour {ip_dca}")
        else:
            print(f"⚠️ Le DCA à {ip_dca} n'a pas répondu favorablement (déjà arrêté ?)")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi de la commande stop à {ip_dca}: {e}")

# Tes configurations
config1 = {
    "ethernetConfig": {
        "systemIPAddress": "192.168.33.30",
        "DCA1000IPAddress": "192.168.33.180",
        "DCA1000ConfigPort": 4096,
        "DCA1000DataPort": 4098,
    }
}

config2 = {
    "ethernetConfig": {
        "systemIPAddress": "192.168.33.32",
        "DCA1000IPAddress": "192.168.33.182",
        "DCA1000ConfigPort": 4099,
        "DCA1000DataPort": 5000,
    }
}

if __name__ == "__main__":
    stop_dca_only(config1)
    stop_dca_only(config2)