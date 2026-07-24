# 8th Server Safety Guard

All remote operations are limited by `ALLOWED_HOSTS` in `configs/deploy_8th.env`.
The deploy precheck requires:

- `TARGET_HOSTS` equals the unique set of configured role hosts plus `BUILD_HOST`.
- `ALLOWED_HOSTS` equals `EXPECTED_HOSTS_CSV`.
- No active endpoint or host appears in `FORBIDDEN_HOSTS_CSV`.

Default forbidden hosts are inherited from the verified 7th migration record as an
example. Adjust the list when the new six-machine pool is chosen.

Runtime writes are limited to this task root, the configured remote containers,
and NFS-mounted paths required by the vLLM service.
