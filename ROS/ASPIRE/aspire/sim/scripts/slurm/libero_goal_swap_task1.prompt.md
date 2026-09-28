# LIBERO-Pro Goal-Swap Task 1 Coordinator

Run only the following ASPIRE Fix Loop task:

- Suite: `libero_goal_swap`
- Task: `open_the_middle_drawer_of_the_cabinet`
- Allocated GPU: logical GPU `0`

Work from `/home/jtangbx/ASPIRE/aspire/sim`.

Read and follow `README.md`, `CLAUDE.md`, `.claude/libero/CLAUDE.md`,
`.claude/libero/fix-loop/INSTRUCTIONS.md`, `.claude/libero/fix-loop/SKILL.md`,
`.claude/libero/fix-loop/main-agent-prompt.md`, and
`.claude/libero/fix-loop/subagent-prompt.md` before acting.

The user explicitly approved these scoped deviations from the reference topology:

- Run only Task 1 rather than all ten tasks.
- Use one Slurm-allocated GPU. SAM3, GraspNet, and LIBERO share logical GPU 0;
  PyRoKi runs on CPU. The batch wrapper has already started and checked the
  perception services.
- The allocation lasts two hours. If prior Task 1 Stage 1 artifacts exist, treat
  them as an interrupted run and continue from them without deleting or replacing
  successful evidence.

Act as the coordinator for this one task. Dispatch exactly one task worker using
the repository's subagent prompt, substituting suite, task, and logical GPU 0.
The worker may use only development seeds 51-65 and must respect the hard limit
of three replay attempts per seed. Do not access held-out seeds 1-50; the batch
wrapper runs Stage 2 after you return. Do not read external baseline code or
outputs, do not access forbidden simulator APIs or asset files, do not touch any
other task, and do not push to a remote.

When the worker finishes, complete the coordinator's Stage 1 promotion flow from
`main-agent-prompt.md`: use only `findings.md` and the selected `fix_code.py`,
record either supported reusable evidence or a documented no-op, and make
`record_skill_promotion.py verify` pass for this suite/task. Stop after Stage 1
and promotion verification. Do not launch held-out validation yourself.
