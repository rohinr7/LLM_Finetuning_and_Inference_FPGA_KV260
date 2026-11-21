# Vitis HLS flow for the milestone 1 matvec kernel (KV260, K26 SOM part).
#
# Run from fpga/hls in a Vitis HLS 2025.2 shell (or any Vitis HLS install):
#   vitis_hls -f run_hls.tcl
#
# Steps: C simulation (runs the host test as testbench), C synthesis
# (latency and resource report), exit. Reports land in matvec_int4_prj/.

open_project -reset matvec_int4_prj
set_top matvec_int4_top
add_files matvec_int4_top.cpp
add_files -tb ../sim/test_matvec_int4.cpp -cflags "-I../hls -std=c++17"

open_solution -reset solution1
set_part {xck26-sfvc784-2LV-c}
create_clock -period 4 -name default   ;# 250 MHz target, as in TeLLMe

csim_design -clean
csynth_design
exit
