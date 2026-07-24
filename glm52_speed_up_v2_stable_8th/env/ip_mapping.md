# 8th IP Mapping

The active topology is configured in `configs/deploy_8th.env`.

Default values mirror the verified 7th deployment and are kept only as an example:

| Role | Config key | 7th example |
| --- | --- | --- |
| Build / image load host | `BUILD_HOST` | `192.168.16.77` |
| Prefill P0 head | `PREFILL0_HEAD` | `192.168.16.63` |
| Prefill P0 worker | `PREFILL0_WORKER` | `192.168.16.65` |
| Prefill P1 head | `PREFILL1_HEAD` | `192.168.16.68` |
| Prefill P1 worker | `PREFILL1_WORKER` | `192.168.16.69` |
| Decode D0 head / Proxy | `DECODE_HEAD` | `192.168.16.76` |
| Decode D0 worker | `DECODE_WORKER` | `192.168.16.77` |

When migrating to a new 6-node pool, update `TARGET_HOSTS`, `ALLOWED_HOSTS`,
and `EXPECTED_HOSTS_CSV` to exactly match these role hosts.
