#!/bin/bash
#sudo modprobe can
#sudo modprobe can_raw
#sudo modprobe mttcan

echo 'nvidia' | sudo -S busybox devmem 0x0c303018 w 0xc458
echo 'nvidia' | sudo -S busybox devmem 0x0c303010 w 0xc400
echo 'nvidia' | sudo -S busybox devmem 0x0c303008 w 0xc458
echo 'nvidia' | sudo -S busybox devmem 0x0c303000 w 0xc400

echo 'nvidia' | sudo -S chmod 666 /sys/devices/platform/bus@0/c310000.mttcan/net/can0/tdc_offset
echo 'nvidia' | sudo -S bash -c "echo 0x600 > /sys/devices/platform/bus@0/c310000.mttcan/net/can0/tdc_offset"
echo 'nvidia' | sudo -S chmod 666 /sys/devices/platform/bus@0/c320000.mttcan/net/can1/tdc_offset
echo 'nvidia' | sudo -S bash -c "echo 0x600 > /sys/devices/platform/bus@0/c320000.mttcan/net/can1/tdc_offset"

echo 'nvidia' | sudo -S ip link set down can0
echo 'nvidia' | sudo -S ip link set can0 type can bitrate 1000000 sample-point 0.875 dbitrate 5000000 dsample-point 0.875 berr-reporting on fd on restart-ms 100
echo 'nvidia' | sudo -S ifconfig can0 txqueuelen 65535
echo 'nvidia' | sudo -S ip link set up can0 mtu 72

#echo 'nvidia' | sudo -S ip link set down can1
#echo 'nvidia' | sudo -S ip link set can1 type can bitrate 1000000 sample-point 0.8 dbitrate 5000000 dsample-point 0.8 berr-reporting on fd on restart-ms 100
#echo 'nvidia' | sudo -S ifconfig can1 txqueuelen 65535
#echo 'nvidia' | sudo -S ip link set up can1 mtu 72

