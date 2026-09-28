# Camera distortion rollback snapshot

Created before testing the company-provided calibration workflow on 2026-07-26.

No robot motion commands are part of this snapshot or test plan.

| Backup file | Original path | SHA-256 |
|---|---|---|
| `body_stereo_ros.yaml` | `/opt/galaxea/body/stereo_ros.yaml` | `195ac1482e7d485cdb94a4a76fb42ed6faf706aefad1edb6e5b836b03701f00a` |
| `body_robot_calibration.json` | `/opt/galaxea/body/robot_calibration.json` | `17dab8de978eab63c63597178b7c3b8d96a185235d46300cf932e5e003ba585c` |
| `sensor_head_left.yaml` | `/opt/galaxea/sensor/camera_calib/head_left.yaml` | `f18a87ae77ec74835b3ec33754368a1ead7d359322b1633d098b3f182b365c92` |
| `sensor_head_right.yaml` | `/opt/galaxea/sensor/camera_calib/head_right.yaml` | `41cd9a3e3a235f47fc66514fc1e7237d37cffb9d5ed73e1eebe21c1dcb3a9044` |
| `signal_camera_head.yaml` | `/home/nvidia/galaxea/install/signal_camera/share/signal_camera/config/signal_camera_head.yaml` | `3b2819fda704d43875ebe970c720672a1c72a5cb91d59c74dd88a6951ad9298a` |
| `data_collection_profile_overrides.yaml` | `/home/nvidia/galaxea/install/data_collection/lib/data_collection/config/default/profile_overrides.yaml` | `cdf436be47c3957729a90322e5118e12f81a8b7ce99bdef5514623d9316244a5` |
| `startup_tools.yaml` | `/home/nvidia/galaxea/install/startup_config/share/startup_config/sessions.d/ATCStandard/R1LITEBody.d/tools.yaml` | `408501752225edd85e70f195b0a3ea02790376f31e34d01ae5534b9a21d709de` |

The preferred rollback is to stop the isolated `jimmy` camera process and restart the unchanged vendor session. Copying files back is only necessary if a later test explicitly modifies one of the originals.
