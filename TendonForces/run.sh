# #!/bin/bash

# python finger.py \
#   --approach_deg 60.0 \
#   --distal_tendon_anchor tip \
#   --data_only \
#   --tension 14.7 \
#   --base_rad 0.01 \
#   --v_mode from_links \
#   --link_lengths "6.6,2.0,2.4,3.0" \
#   --joint_lengths "2.0,2.0,2.0" \
#   --joint_E "0.1,0.08,0.06" \
#   --E 6.74e6 \
#   --damping 0.1 \
#   --n_elements 0 \
#   --final_time 2.0 \
#   --cyl_rad 0.025 \
#   --k_contact 500.0 \
#   --max_penetration_warn 0.002 \
#   --base_len 0.20 \
#   --cross_section rect \
#   --base_width 0.03 \
#   --base_thickness 0.018 \
#   --nu_contact 20.0 \
#   --mu_contact 0.8 \
#   --vel_damp_contact 30 \
#   --poisson_nu 0.49 \
#   --v_mass 0.04 \
#   --num_v 3 \
#   --v_start 38 \
#   --v_end 80 \
#   --v_height 0.002 \
#   --body_mass 3 \
#   --suffix manufactured_real_from_runsh \
#   --output_dir debug_runs \
#   --landing_motion \
#   --landing_mode prescribed \
#   --ankle_wrap_radius 0.03 \
#   --ankle_stiffness 500.0 \
#   --min_tension 0.1 \
#   --max_tension 20.0 \
#   --landing_speed 0.0 \
#   --initial_x_gap 0.15 \
#   --landing_height 0.04 \
#   --landing_approach_deg 30.0 \
#   --prescribed_stop_at_contact \
#   --prescribed_contact_margin -0.005 \
#   --base_force_mag 0.0 \
#   --base_force_dir "0,0,-1" \
#   --base_force_nodes 1 \
#   --force_driven_stabilize \
#   --force_driven_xy_k 120.0 \
#   --force_driven_xy_c 3.0 \
#   --force_driven_tendon_ramp 1.0 \
#   --force_driven_xy_fmax 5.0 \
#   --force_driven_lock_base_xy \
#   --force_driven_z_stabilize \
#   --force_driven_z_k 120.0 \
#   --force_driven_z_c 12.0 \
#   --force_driven_z_fmax 4.0 \
#   --force_driven_z_target cylinder \
#   --force_driven_z_target_offset -0.01 \
#   --force_driven_min_damping 5.0 \
#   --force_driven_node_drag 4.0 \
#   --force_driven_node_drag_axes "1,1,1" \
#   --force_driven_rot_stabilize \
#   --force_driven_rot_k 0.03 \
#   --force_driven_rot_c 0.02 \
#   --force_driven_rot_tmax 0.02 \
#   --disturbance_force_mag 5.0 \
#   --disturbance_base_nodes 5 \
#   --disturbance_steps 100 \
#   --disturbance_dt_scale 1.0 \
#   --continuous_disturbance_metric \
#   --joint_stiffness_mode bending_only \

#!/bin/bash

python finger.py \
  --approach_deg 65.0 \
  --distal_tendon_anchor tip \
  --tension 2.5 \
  --base_rad 0.0115 \
  --v_mode from_links \
  --link_lengths "6.6,2.0,2.4,3.0" \
  --joint_lengths "2.0,2.0,2.0" \
  --joint_E "0.1,0.08,0.06" \
  --E 6.74e6 \
  --damping 0.1 \
  --n_elements 0 \
  --final_time 2.0 \
  --cyl_rad 0.025 \
  --k_contact 500.0 \
  --max_penetration_warn 0.002 \
  --base_len 0.20 \
  --cross_section rect \
  --base_width 0.03 \
  --base_thickness 0.018 \
  --nu_contact 20.0 \
  --mu_contact 0.8 \
  --vel_damp_contact 30 \
  --poisson_nu 0.49 \
  --v_mass 0.04 \
  --num_v 3 \
  --v_start 38 \
  --v_end 80 \
  --v_height 0.002 \
  --body_mass 3 \
  --suffix debug_unstable_T2p5_A65 \
  --output_dir debug_runs \
  --landing_motion \
  --landing_mode prescribed \
  --ankle_wrap_radius 0.0225 \
  --ankle_stiffness 700.0 \
  --min_tension 0.1 \
  --max_tension 20.0 \
  --landing_speed 0.0 \
  --initial_x_gap 0.15 \
  --landing_height 0.04 \
  --landing_approach_deg 30.0 \
  --prescribed_stop_at_contact \
  --prescribed_contact_margin -0.005 \
  --base_force_mag 0.0 \
  --base_force_dir "0,0,-1" \
  --base_force_nodes 1 \
  --force_driven_stabilize \
  --force_driven_xy_k 120.0 \
  --force_driven_xy_c 3.0 \
  --force_driven_tendon_ramp 1.0 \
  --force_driven_xy_fmax 5.0 \
  --force_driven_lock_base_xy \
  --force_driven_z_stabilize \
  --force_driven_z_k 120.0 \
  --force_driven_z_c 12.0 \
  --force_driven_z_fmax 4.0 \
  --force_driven_z_target cylinder \
  --force_driven_z_target_offset -0.01 \
  --force_driven_min_damping 5.0 \
  --force_driven_node_drag 4.0 \
  --force_driven_node_drag_axes "1,1,1" \
  --force_driven_rot_stabilize \
  --force_driven_rot_k 0.03 \
  --force_driven_rot_c 0.02 \
  --force_driven_rot_tmax 0.02 \
  --disturbance_force_mag 5.0 \
  --disturbance_base_nodes 5 \
  --disturbance_steps 100 \
  --disturbance_dt_scale 1.0 \
  --continuous_disturbance_metric \
  --joint_stiffness_mode bending_only