
export ENCODER_BITRATE_PERIOD_MS=100

stdbuf --output=L ./sunshine | tee -i x
#gdb --args ./sunshine
