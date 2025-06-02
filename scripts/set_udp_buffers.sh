#!/bin/bash
# This script makes UDP buffer size changes permanent for Cloudflared and other apps.
# Run as root (sudo)!

CONF_FILE="/etc/sysctl.conf"

# Add or update the required sysctl settings
if grep -q '^net.core.rmem_max' "$CONF_FILE"; then
    sudo sed -i 's/^net.core.rmem_max.*/net.core.rmem_max=2500000/' "$CONF_FILE"
else
    echo 'net.core.rmem_max=2500000' | sudo tee -a "$CONF_FILE"
fi

if grep -q '^net.core.rmem_default' "$CONF_FILE"; then
    sudo sed -i 's/^net.core.rmem_default.*/net.core.rmem_default=2500000/' "$CONF_FILE"
else
    echo 'net.core.rmem_default=2500000' | sudo tee -a "$CONF_FILE"
fi

# Apply changes immediately
sudo sysctl -p

echo "UDP buffer size settings applied and made permanent."
