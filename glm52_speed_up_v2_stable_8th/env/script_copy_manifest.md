# 8th Script Manifest

Source deployment script:

```text
glm52_speed_up_v2_stable_7th/scripts/deploy/deploy_7th_claude204_current_verified.sh
```

Primary 8th entrypoints:

```text
scripts/deploy/deploy_8th_claude204_current_verified.sh
scripts/deploy/deploy_8th_current_verified.sh
scripts/deploy/deploy_stage_90k_v8_prefix_cache.sh
scripts/deploy/deploy_stage_90k_v2.sh
scripts/deploy/deploy_2p1d_from_image.sh
scripts/deploy/lib/manage_stack_2p1d.sh
scripts/deploy/lib/container_entry.sh
scripts/watchdog/prefill_decode_availability_watchdog.sh
```

Compatibility entrypoints retained in the 8th folder:

```text
scripts/deploy/deploy_7th_claude204_current_verified.sh -> deploy_8th_claude204_current_verified.sh
scripts/deploy/deploy_7th_current_verified.sh -> deploy_8th_current_verified.sh
scripts/deploy/deploy_stage_90k_v7_prefix_cache.sh -> deploy_8th_current_verified.sh
```

Copied but not copied-forward as active service-watchdog flow:

```text
The old watchdog_common/install/service/final_drill chain was removed from 8th.
Use scripts/watchdog/prefill_decode_availability_watchdog.sh through the main
deploy wrapper instead.
```
