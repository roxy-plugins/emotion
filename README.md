# emotion

Akashic emotion and proactive tuning plugin.

`feedback-preference-context` 只把反馈归纳结果追加到 workspace 根目录的
`proactive_pending.md`。该队列由用户或明确授权的维护流程审核；插件不会在 Drift
完成后自动改写 `PROACTIVE_CONTEXT.md`。

## 移动端看板

插件通过通用移动 UI 生命周期注册“主动状态”入口，说明用户反馈如何改变 Agent 的语气
与主动发送把握。移动端只列真正产生状态增量的反馈，不复制桌面端每个 proactive tick
的 effect 表；原始 VAD 指标默认折叠，需要时再查看。
