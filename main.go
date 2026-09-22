package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// ==================== 1. 数据结构定义 ====================

// ---------- 本地接收到的 Chat Completions 请求 ----------

type ChatFunctionDef struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

type ChatTool struct {
	Type     string           `json:"type"`
	Function *ChatFunctionDef `json:"function,omitempty"`
}

type ChatToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ChatToolCall 同时用于最终消息与流式 delta(流式时携带 index)
type ChatToolCall struct {
	Index    int                   `json:"index,omitempty"`
	ID       string                `json:"id,omitempty"`
	Type     string                `json:"type,omitempty"`
	Function *ChatToolCallFunction `json:"function,omitempty"`
}

type ChatMessage struct {
	Role       string         `json:"role"`
	Content    interface{}    `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []ChatToolCall `json:"tool_calls,omitempty"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Tools       []ChatTool    `json:"tools,omitempty"`
	ToolChoice  interface{}   `json:"tool_choice,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

// ---------- 返回给客户端的 Chat Completions 响应 ----------

type ChatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"` // "chat.completion"
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *ChatUsage             `json:"usage,omitempty"`
}

// 流式 chunk
type ChatChunkDelta struct {
	Role      string         `json:"role,omitempty"`
	Content   *string        `json:"content,omitempty"`
	ToolCalls []ChatToolCall `json:"tool_calls,omitempty"`
}

type ChatChunkChoice struct {
	Index        int            `json:"index"`
	Delta        ChatChunkDelta `json:"delta"`
	FinishReason *string        `json:"finish_reason"` // 流式中平时为 null
}

type ChatCompletionChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"` // "chat.completion.chunk"
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ChatChunkChoice `json:"choices"`
}

// ---------- 发给上游的 Responses API 请求 ----------

type ResponsesTool struct {
	Type        string      `json:"type"`
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

// ResponsesInputItem 覆盖 message / function_call / function_call_output 三种输入项
type ResponsesInputItem struct {
	Type      string      `json:"type,omitempty"` // message 项可省略,仅留 role
	Role      string      `json:"role,omitempty"`
	Content   interface{} `json:"content,omitempty"`
	CallID    string      `json:"call_id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Arguments string      `json:"arguments,omitempty"`
	Output    string      `json:"output,omitempty"`
}

type ResponsesRequest struct {
	Model           string               `json:"model"`
	Instructions    string               `json:"instructions,omitempty"`
	Input           []ResponsesInputItem `json:"input"`
	Tools           []ResponsesTool      `json:"tools,omitempty"`
	ToolChoice      interface{}          `json:"tool_choice,omitempty"`
	Temperature     *float64             `json:"temperature,omitempty"`
	TopP            *float64             `json:"top_p,omitempty"`
	MaxOutputTokens *int                 `json:"max_output_tokens,omitempty"`
	Stream          bool                 `json:"stream,omitempty"`
}

// ---------- 上游 Responses API 的非流式响应 ----------

type ResponsesContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type ResponsesOutputItem struct {
	Type      string                 `json:"type"` // "message" | "function_call" | ...
	ID        string                 `json:"id,omitempty"`
	Role      string                 `json:"role,omitempty"`
	Content   []ResponsesContentPart `json:"content,omitempty"`
	CallID    string                 `json:"call_id,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Arguments string                 `json:"arguments,omitempty"`
}

type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type ResponsesAPIResponse struct {
	ID     string                `json:"id"`
	Object string                `json:"object"`
	Model  string                `json:"model"`
	Output []ResponsesOutputItem `json:"output"`
	Usage  *ResponsesUsage       `json:"usage,omitempty"`
}

// ---------- 上游 Responses API 的流式事件 ----------

type ResponsesStreamEvent struct {
	Type        string                `json:"type"`
	OutputIndex int                   `json:"output_index"`
	ItemID      string                `json:"item_id"`
	Delta       string                `json:"delta"`
	Item        *ResponsesOutputItem  `json:"item"`
	Response    *ResponsesAPIResponse `json:"response"`
}

// ==================== 2. 工具函数 ====================

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// joinResponsesURL: 兼容 base 带/不带 /v1 两种配置,拼出最终的 /v1/responses 地址
func joinResponsesURL(base string) string {
	base = strings.TrimSuffix(base, "/")
	if strings.HasSuffix(base, "/v1") {
		return base + "/responses"
	}
	return base + "/v1/responses"
}

// contentToString: 把 Chat Completions 的 content(string 或多模态数组)压成纯文本
func contentToString(c interface{}) string {
	switch v := c.(type) {
	case string:
		return v
	case []interface{}:
		var sb strings.Builder
		for _, part := range v {
			if m, ok := part.(map[string]interface{}); ok {
				if t, _ := m["text"].(string); t != "" {
					sb.WriteString(t)
				}
			}
		}
		return sb.String()
	}
	return ""
}

func cloneMap(m map[string]interface{}) map[string]interface{} {
	cp := make(map[string]interface{}, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}

// convertUserContent: 将 Chat Completions 多模态数组映射为 Responses 输入格式
// text -> input_text, image_url -> input_image, 其余原样透传
func convertUserContent(c interface{}) interface{} {
	arr, ok := c.([]interface{})
	if !ok {
		return c // 字符串等,原样透传
	}
	out := make([]interface{}, 0, len(arr))
	changed := false
	for _, item := range arr {
		m, ok := item.(map[string]interface{})
		if !ok {
			out = append(out, item)
			continue
		}
		switch t, _ := m["type"].(string); t {
		case "text":
			cp := cloneMap(m)
			cp["type"] = "input_text"
			out = append(out, cp)
			changed = true
		case "image_url":
			cp := cloneMap(m)
			cp["type"] = "input_image"
			out = append(out, cp)
			changed = true
		default: // input_text/input_file 等已符合 Responses 格式,直接透传
			out = append(out, m)
		}
	}
	if !changed {
		return c
	}
	return out
}

// ==================== 3. 请求转换: Chat Completions -> Responses ====================

func convertChatToResponsesReq(chatReq *ChatCompletionRequest) *ResponsesRequest {
	var systemParts []string
	input := make([]ResponsesInputItem, 0, len(chatReq.Messages)*2)

	for _, msg := range chatReq.Messages {
		switch msg.Role {
		case "system", "developer":
			// system prompt 全部合并进 Responses 的 instructions
			if s := contentToString(msg.Content); s != "" {
				systemParts = append(systemParts, s)
			}
		case "assistant":
			// 文本部分 -> message 输入项
			if s := contentToString(msg.Content); s != "" {
				input = append(input, ResponsesInputItem{Role: "assistant", Content: s})
			}
			// 历史工具调用 -> function_call 输入项
			for _, tc := range msg.ToolCalls {
				if tc.Function == nil {
					continue
				}
				input = append(input, ResponsesInputItem{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				})
			}
		case "tool":
			// 工具执行结果 -> function_call_output 输入项
			input = append(input, ResponsesInputItem{
				Type:   "function_call_output",
				CallID: msg.ToolCallID,
				Output: contentToString(msg.Content),
			})
		default: // user
			input = append(input, ResponsesInputItem{
				Role:    msg.Role,
				Content: convertUserContent(msg.Content),
			})
		}
	}

	// tools: {type:function, function:{...}} -> 扁平的 {type:function, name, ...}
	var tools []ResponsesTool
	for _, t := range chatReq.Tools {
		if t.Type == "function" && t.Function != nil {
			tools = append(tools, ResponsesTool{
				Type:        "function",
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
			})
		}
	}

	return &ResponsesRequest{
		Model:           chatReq.Model,
		Instructions:    strings.Join(systemParts, "\n"),
		Input:           input,
		Tools:           tools,
		ToolChoice:      convertToolChoice(chatReq.ToolChoice),
		Temperature:     chatReq.Temperature,
		TopP:            chatReq.TopP,
		MaxOutputTokens: chatReq.MaxTokens, // max_tokens -> max_output_tokens
		Stream:          chatReq.Stream,
	}
}

// convertToolChoice: "auto"/"none"/"required" 原样透传;
// {type:function, function:{name}} -> {type:function, name}
func convertToolChoice(tc interface{}) interface{} {
	switch v := tc.(type) {
	case string:
		return v
	case map[string]interface{}:
		if fn, ok := v["function"].(map[string]interface{}); ok {
			if name, ok := fn["name"].(string); ok {
				return map[string]interface{}{"type": "function", "name": name}
			}
		}
	}
	return nil
}

// ==================== 4. HTTP Handler ====================

// 上游客户端:不设整体 Timeout(避免掐断长流式),用 r.Context() 控制生命周期
var upstreamClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   32,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: 10 * time.Minute,
	},
}

func handleChatToResponsesProxy(responsesURL, upstreamAPIKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// 简单 CORS 支持
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 1. 读取并解析本地发送的 Chat Completions 请求
		var chatReq ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
			http.Error(w, fmt.Sprintf("Invalid Chat Completions payload: %v", err), http.StatusBadRequest)
			return
		}

		// 2. 转换为 Responses API 请求结构
		respReq := convertChatToResponsesReq(&chatReq)

		// 3. 构建发往上游 /v1/responses 的请求
		respPayload, _ := json.Marshal(respReq)
		req, err := http.NewRequestWithContext(r.Context(), "POST", responsesURL, bytes.NewReader(respPayload))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		req.Header.Set("Content-Type", "application/json")
		if chatReq.Stream {
			req.Header.Set("Accept", "text/event-stream")
		}
		// 鉴权:优先使用代理配置的 UPSTREAM_API_KEY,否则透传客户端的 Authorization 头
		if upstreamAPIKey != "" {
			req.Header.Set("Authorization", "Bearer "+upstreamAPIKey)
		} else if auth := r.Header.Get("Authorization"); auth != "" {
			req.Header.Set("Authorization", auth)
		}
		if key := r.Header.Get("Api-Key"); key != "" {
			req.Header.Set("Api-Key", key)
		}

		resp, err := upstreamClient.Do(req)
		if err != nil {
			http.Error(w, fmt.Sprintf("Upstream request failed: %v", err), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			// 上游报错原样输出
			w.WriteHeader(resp.StatusCode)
			io.Copy(w, resp.Body)
			return
		}

		// 4. 按流式/非流式分别转换响应
		if chatReq.Stream {
			handleStreamResponse(w, resp.Body, chatReq.Model)
		} else {
			handleJSONResponse(w, resp.Body, chatReq.Model)
		}
	}
}

// ==================== 5. 非流式响应转换: Responses -> Chat Completions ====================

func handleJSONResponse(w http.ResponseWriter, upstreamBody io.Reader, reqModel string) {
	var out ResponsesAPIResponse
	if err := json.NewDecoder(upstreamBody).Decode(&out); err != nil {
		http.Error(w, fmt.Sprintf("Failed to parse upstream Responses payload: %v", err), http.StatusBadGateway)
		return
	}

	// 提取 assistant 文本与工具调用
	var textParts []string
	var toolCalls []ChatToolCall
	finishReason := "stop"
	for _, item := range out.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					textParts = append(textParts, part.Text)
				}
			}
		case "function_call":
			toolCalls = append(toolCalls, ChatToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: &ChatToolCallFunction{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		}
	}
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	var content interface{}
	if len(textParts) > 0 {
		content = strings.Join(textParts, "")
	} // 纯工具调用时保持 nil -> "content": null

	id := out.ID
	if id == "" {
		id = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}

	var usage *ChatUsage
	if out.Usage != nil {
		total := out.Usage.TotalTokens
		if total == 0 {
			total = out.Usage.InputTokens + out.Usage.OutputTokens
		}
		usage = &ChatUsage{
			PromptTokens:     out.Usage.InputTokens,
			CompletionTokens: out.Usage.OutputTokens,
			TotalTokens:      total,
		}
	}

	chatResp := ChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   reqModel,
		Choices: []ChatCompletionChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:      "assistant",
					Content:   content,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(chatResp)
}

// ==================== 6. 流式响应转换: Responses SSE -> Chat Completions SSE ====================

func handleStreamResponse(w http.ResponseWriter, upstreamBody io.Reader, reqModel string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	respID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	sendChunk := func(delta ChatChunkDelta, finish *string) {
		chunk := ChatCompletionChunk{
			ID:      respID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   reqModel,
			Choices: []ChatChunkChoice{{Index: 0, Delta: delta, FinishReason: finish}},
		}
		if b, err := json.Marshal(chunk); err == nil {
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
	}

	sendDone := func() {
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}

	scanner := bufio.NewScanner(upstreamBody)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024) // 单行最大 8MB,防长输出被截断

	sentRole := false
	finished := false
	// item_id -> 本地 tool_calls 下标 的映射(用于参数增量回填)
	toolIndexByID := map[string]int{}

	sendRoleIfNeeded := func() {
		if !sentRole {
			sentRole = true
			sendChunk(ChatChunkDelta{Role: "assistant"}, nil)
		}
	}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var ev ResponsesStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}

		switch ev.Type {
		case "response.output_item.added":
			// 新输出项:文本消息先发 role;函数调用则发工具调用的头部(id + name)
			if ev.Item == nil {
				continue
			}
			switch ev.Item.Type {
			case "message":
				sendRoleIfNeeded()
			case "function_call":
				idx := len(toolIndexByID)
				toolIndexByID[ev.Item.ID] = idx
				sendChunk(ChatChunkDelta{
					ToolCalls: []ChatToolCall{{
						Index: idx,
						ID:    ev.Item.CallID,
						Type:  "function",
						Function: &ChatToolCallFunction{
							Name: ev.Item.Name,
						},
					}},
				}, nil)
			}

		case "response.output_text.delta":
			// 文本增量
			if ev.Delta == "" {
				continue
			}
			if !sentRole {
				sentRole = true
				sendChunk(ChatChunkDelta{Role: "assistant", Content: &ev.Delta}, nil)
			} else {
				sendChunk(ChatChunkDelta{Content: &ev.Delta}, nil)
			}

		case "response.function_call_arguments.delta":
			// 工具参数增量
			if ev.Delta == "" {
				continue
			}
			idx, ok := toolIndexByID[ev.ItemID]
			if !ok {
				idx = ev.OutputIndex // 兜底:用上游 output_index
				toolIndexByID[ev.ItemID] = idx
			}
			sendChunk(ChatChunkDelta{
				ToolCalls: []ChatToolCall{{
					Index:    idx,
					Function: &ChatToolCallFunction{Arguments: ev.Delta},
				}},
			}, nil)

		case "response.completed":
			// 结束:根据是否有 function_call 决定 finish_reason
			finishReason := "stop"
			if ev.Response != nil {
				for _, item := range ev.Response.Output {
					if item.Type == "function_call" {
						finishReason = "tool_calls"
						break
					}
				}
			}
			finished = true
			sendChunk(ChatChunkDelta{}, &finishReason)
			sendDone()
			return

		case "response.failed", "error":
			// 上游失败:以 OpenAI 流式错误格式通知客户端后结束
			msg := data
			if ev.Response != nil {
				msg = fmt.Sprintf("upstream responses api failed: %s", ev.Type)
			}
			errPayload, _ := json.Marshal(map[string]interface{}{
				"error": map[string]string{
					"message": msg,
					"type":    "upstream_error",
				},
			})
			fmt.Fprintf(w, "data: %s\n\n", errPayload)
			flusher.Flush()
			finished = true
			sendDone()
			return
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("读取上游 SSE 出错: %v", err)
	}

	// 上游流提前结束(未收到 response.completed)也要正常收尾,避免客户端挂起
	if !finished {
		stop := "stop"
		sendChunk(ChatChunkDelta{}, &stop)
		sendDone()
	}
}

// ==================== 7. 主函数 ====================

func main() {
	// 上游只支持 Responses API 的服务器地址(通过环境变量 UPSTREAM_BASE_URL 配置)
	// 兼容 http://host:9695 与 http://host:9695/v1 两种写法
	upstreamBaseURL := envOr("UPSTREAM_BASE_URL", "http://172.20.156.12:9695/v1")
	responsesURL := joinResponsesURL(upstreamBaseURL)
	upstreamAPIKey := os.Getenv("UPSTREAM_API_KEY") // 可选:代理统一注入密钥
	port := envOr("PORT", "8081")
	if !strings.HasPrefix(port, ":") {
		port = ":" + port // 兼容 PORT=8081 与 PORT=:8081 两种写法
	}

	proxy := handleChatToResponsesProxy(responsesURL, upstreamAPIKey)

	// 同时注册带/不带 /v1 前缀的端点,兼容各类客户端
	http.HandleFunc("/v1/chat/completions", proxy)
	http.HandleFunc("/chat/completions", proxy)

	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})

	log.Printf("代理已启动!本地 Chat Completions 地址: http://localhost%s/v1/chat/completions", port)
	log.Printf("请求将被自动转换为上游 Responses API: %s", responsesURL)
	log.Printf("提示: 可通过环境变量 UPSTREAM_BASE_URL / PORT 修改上游地址与监听端口")

	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("服务启动失败: %v", err)
	}
}
