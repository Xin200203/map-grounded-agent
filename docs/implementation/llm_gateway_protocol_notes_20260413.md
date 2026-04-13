# SmoothNav LLM Gateway Protocol Notes (2026-04-13)

## 为什么要写这份说明

本次在 `clauddy.com` 上接 SmoothNav 时，出现了一个容易反复踩坑的问题：

- 本地 `Claude Code` 可以正常对话
- 但项目里手写的 `OpenAI(...).chat.completions.create(...)` 调用失败

结论不是“账号一定坏了”，而是：

**同一个 Clauddy 账号下，不同模型族和不同客户端，可能走的是不同协议。**

---

## 本次确认到的关键事实

### 1. Claude 系列不应默认走 OpenAI chat-completions

对接 Clauddy 时，`Claude` 系列更接近：

- `provider = anthropic`
- `protocol = anthropic-messages`
- `base_url = https://clauddy.com`
- endpoint: `/v1/messages`

这和本地 `Claude Code` 的调试日志一致：

- 使用 `ANTHROPIC_BASE_URL=https://clauddy.com`
- 请求路径是 `/v1/messages`

### 2. GPT / Codex 系列应按 OpenAI 兼容协议处理

对接 Clauddy 时，`GPT / Codex` 系列更接近：

- `provider = openai`
- `protocol = openai-responses`
- `base_url = https://clauddy.com/v1`
- endpoint: `/responses`

不要再把所有模型都强行塞进 `/chat/completions`。

### 3. “同账号”不等于“同协议可互换”

即使多个 key 最终都映射到同一个 Clauddy 账号，也不能推出：

- Claude Code 的 `oauth_token` 路径
- OpenAI-compatible `/chat/completions`
- OpenAI-compatible `/responses`
- Anthropic-native `/v1/messages`

一定完全等价。

需要分别验证：

- 请求协议是否匹配模型族
- token group 是否覆盖目标模型
- 当前 token 是否能走对应网关

---

## 当前项目的实现约定

从 2026-04-13 开始，SmoothNav 增加两项显式配置：

- `api_provider`
- `api_protocol`

当前支持：

- `anthropic + anthropic-messages`
- `openai + openai-responses`
- `openai + openai-chat-completions`（仅保留为 legacy fallback）

默认值：

- `api_provider: anthropic`
- `api_protocol: anthropic-messages`

原因：

- 当前默认模型配置是 Claude family
- 与 Clauddy 文档和本地 Claude Code 观察结果更一致

---

## 当前推荐配置

### Claude via Clauddy

```yaml
api_provider: "anthropic"
api_protocol: "anthropic-messages"
base_url_env: "SMOOTHNAV_BASE_URL"
api_key_env: "SMOOTHNAV_API_KEY"
llm_model: "claude-sonnet-4-5-20250929"
llm_model_fast: "claude-haiku-4-5-20251001"
vlm_model: "claude-haiku-4-5-20251001"
```

环境变量：

```bash
export SMOOTHNAV_BASE_URL="https://clauddy.com"
export SMOOTHNAV_API_KEY="..."
```

### GPT / Codex via Clauddy

```yaml
api_provider: "openai"
api_protocol: "openai-responses"
base_url_env: "SMOOTHNAV_BASE_URL"
api_key_env: "SMOOTHNAV_API_KEY"
```

环境变量：

```bash
export SMOOTHNAV_BASE_URL="https://clauddy.com/v1"
export SMOOTHNAV_API_KEY="..."
```

---

## 排障顺序

如果后面再次出现 “Claude Code 能用，但项目不能用”，按这个顺序排：

1. 先确认项目配置里的 `api_provider/api_protocol` 是否和模型族匹配。
2. 再确认 `base_url` 是否与协议匹配。
3. 再做最小请求验证：
   - Claude: `/v1/messages`
   - GPT/Codex: `/v1/responses`
4. 最后再看 token group、额度、风控、出口网络。

不要一上来就只盯着：

- 余额
- 单个 key
- `chat.completions`

因为问题很可能根本不在这三项。

---

## 本地观察记录

本地 `Claude Code` 当前已确认：

- auth method: `oauth_token`
- api provider: `firstParty`
- 配置中使用 `ANTHROPIC_BASE_URL=https://clauddy.com`
- debug log 显示请求走 `/v1/messages`

这说明：

**Claude Code 可用，不能自动推出 OpenAI-compatible chat-completions 也可用。**

---

## 2026-04-13 补充观察

今天又确认了一层更细的差异：

- 通过官方 `anthropic` SDK 调 `https://clauddy.com`
- 使用本地 `Claude Code` 那套 `ANTHROPIC_AUTH_TOKEN`
- `claude-haiku-4-5-20251001` 可以正常返回文本内容

但如果改用当前实验里提供的 `sk-...` key：

- `anthropic` SDK 请求本身可以成功
- 但返回对象会出现异常现象：
  - 请求的是 Claude 模型名
  - 返回里的 `model` 却可能变成 `gpt-5.1-codex-mini`
  - `content` 为空列表

这说明后续排障不能只看“HTTP 是否 200”：

1. 还要检查响应里实际落到的模型名。
2. 还要检查 `content/output` 是否真的非空。
3. 同一个 Clauddy 账号下，不同 token 很可能挂在不同分组或不同网关映射上。

因此当前项目的工程策略是：

- `anthropic-messages` 在工程实现上优先复用 Anthropic 风格 headers，但实际请求链路改为 `httpx -> https://clauddy.com/v1/messages`
- `openai-responses` 继续保留为 OpenAI 兼容路线
- 实验前必须做一次最小文本返回验证，不能只测连通性

---

## 当前工程实现细节（已用于真实实验）

### 1. `anthropic-messages` 的稳定实现

当前 [llm.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/utils/llm.py) 中，`anthropic-messages` 不再走：

- 旧版手写 `urllib` 请求
- 服务器上的旧版 `anthropic` SDK `messages.create(...)`

当前稳定路径是：

1. 用项目配置解析出：
   - `provider = anthropic`
   - `protocol = anthropic-messages`
   - `base_url = https://clauddy.com`
2. 通过 `_endpoint_for_protocol(...)` 解析成：
   - `https://clauddy.com/v1/messages`
3. 用 `Anthropic(...)` client 生成与 SDK 一致的：
   - `default_headers`
   - `auth_headers`
4. 用 `httpx.Client(...).post(...)` 发送 JSON 请求

这样做的原因很直接：

- Clauddy 对纯 `urllib` 裸请求会返回 `403 / error code: 1010`
- 服务器上的 `anthropic==0.72.0` 会把 `base_url=https://clauddy.com` 处理成 `https://clauddy.com/v1/`
- 再叠加 `messages.create()` 内部固定的 `"/v1/messages"`，最终就会打成错误的 `/v1/v1/messages`

所以当前最稳的组合不是“完全手写”，也不是“完全依赖旧 SDK”，而是：

**Anthropic 风格 headers + `httpx` 直接请求 Clauddy 的正确 endpoint。**

### 2. 哪些路径已经验证过

已经实际验证过的路径如下：

- 本地 `Claude Code` 自带 token + Anthropic SDK：可返回正常文本
- 本地 `Anthropic headers + httpx + /v1/messages`：可返回正常文本
- 184 服务器 `Anthropic headers + httpx + /v1/messages`：可返回正常文本
- 184 服务器 `baseline text 5`：已进入真实 episode 采样并持续写 `step_traces`

已知不稳定或错误的路径：

- `urllib` 直接 POST Clauddy `/v1/messages`：被边缘层拦成 `1010`
- 服务器旧版 `anthropic==0.72.0` 直接 `messages.create()`：会出现 `/v1/v1/messages`
- 某些 `sk-...` token 虽然返回 `200`，但可能：
  - 响应模型被映射成别的模型
  - `content` 为空

因此工程上必须把“`HTTP 200`”和“真的有有效文本输出”分开看。

### 3. 当前推荐的运行顺序

如果使用仓库内的 `run.sh`，当前推荐方式是：

1. 激活 conda 环境
2. 进入项目根目录
3. 让 `run.sh` 自动读取 git 忽略的本地凭据文件 `.local/clauddy.env.sh`
4. 再启动实验

`run.sh` 现在会自动 source：

```bash
<repo>/.local/clauddy.env.sh
```

这个文件应至少包含：

```bash
export SMOOTHNAV_BASE_URL="https://clauddy.com"
export SMOOTHNAV_API_KEY="..."
```

文件是**本地明文、但 git 忽略**的，适合实验机和开发机长期复用。

### 4. 为什么要让本地凭据文件覆盖环境变量

这次实验里已经遇到一个真实坑：

- `conda activate unigoal` 可能会通过 `activate.d` 注入旧的 `SMOOTHNAV_API_KEY`
- 如果后面不再覆盖，运行时就会悄悄回到错误 token

因此当前设计是：

- `run.sh` 在启动前再 source 一次本地 `.local/clauddy.env.sh`
- 让 repo-local 凭据文件覆盖 shell/conda 中的旧值

这一步是为了保证：

**真正执行实验时使用的是当前确认可用的 Clauddy 凭据，而不是历史遗留环境变量。**

---

## 外部参考

- Clauddy Claude Code 配置: `https://docs.clauddy.com/cli/claude-code.html`
- Clauddy Codex 配置: `https://docs.clauddy.com/cli/codex.html`
- Clauddy OpenClaw 协议说明: `https://docs.clauddy.com/advanced/openclaw`
- Clauddy 模型广场 / 计费说明: `https://docs.clauddy.com/models/marketplace.html`
- Clauddy 令牌分组说明: `https://docs.clauddy.com/models/token-groups.html`
