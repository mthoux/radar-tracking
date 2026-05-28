
#for 1 people 
cfg_gtrack = GTrackConfig2D(
        max_points=200,
        max_tracks=1,
        dt=0.6,
        process_noise=0.01,              
        meas_noise_range=2.0,           
        meas_noise_az=0.5,              
        gating_threshold=6,
        alloc_range_gate=0.8,           
        alloc_az_gate=np.deg2rad(15),    
        alloc_vel_gate=20,
        min_cluster_points=10,
        alloc_snr_threshold=0.5,
        min_snr_threshold=0.005,
        signal_threshold=0.3,  # Minimum normalized signal for a point (0-1)
        init_state_cov=1.0,
        det_to_active_count=3,          
        det_to_free_count=6,
        act_to_free_count=8,
        presence_zones=[],
        pres_on_count=5,
        pres_off_count=3
    )

#for 2 people 
cfg_gtrack = GTrackConfig2D(
        max_points=200,
        max_tracks=2,
        dt=0.6,
        process_noise=0.05,              
        meas_noise_range=0.5,           
        meas_noise_az=0.05,              
        gating_threshold=6,
        alloc_range_gate=0.5,           
        alloc_az_gate=np.deg2rad(10),   
        alloc_vel_gate=20,
        min_cluster_points=10,
        alloc_snr_threshold=0.5,
        min_snr_threshold=0.005,
        signal_threshold=0.4,  # Minimum normalized signal for a point (0-1)
        init_state_cov=1.0,
        det_to_active_count=3,          
        det_to_free_count=6,
        act_to_free_count=8,
        presence_zones=[],
        pres_on_count=5,
        pres_off_count=3
    )

#for 3 people 
    cfg_gtrack = GTrackConfig2D(
        max_points=200,
        max_tracks=3,
        dt=0.6,
        process_noise=0.05,             
        meas_noise_range=0.5,           
        meas_noise_az=0.05,             
        gating_threshold=3,
        alloc_range_gate=0.5,           
        alloc_az_gate=np.deg2rad(7),    
        alloc_vel_gate=20,
        min_cluster_points=6,
        alloc_snr_threshold=0.5,
        min_snr_threshold=0.005,
        signal_threshold=0.6,  
        init_state_cov=1.0,
        det_to_active_count=3,
        det_to_free_count=6,
        act_to_free_count=8,
        presence_zones=[],
        pres_on_count=5,
        pres_off_count=3
    )