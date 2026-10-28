---
changeKind: fix
packages:
  - sdk/agentserver/azure-ai-agentserver-responses
---

Persist resilient background checkpoints with the same resolved response, agent,
model, session, conversation, and background identity as terminal snapshots,
including streams constructed without a request seed. Export the existing
`ResponseCheckpointEvent` control-event type from the package and `streaming`
namespace so hosts no longer need private imports to recognize it.
