package main

import (
	"bufio"
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"maps"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// ==================== 1. 数据结构定义 ====================

// ---------- 本地接收到的 Chat Completions 请求 ----------

type ChatFunctionDef struct {
	Strict      *bool       `json:"strict,omitempty"`
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
	Index    *int                  `json:"index,omitempty"`
	ID       string                `json:"id,omitempty"`
	Type     string                `json:"type,omitempty"`
	Function *ChatToolCallFunction `json:"function,omitempty"`
}

type ChatMessage struct {
	Role       string         `json:"role"`
	Content    interface{}    `json:"content"`
	Refusal    string         `json:"refusal,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []ChatToolCall `json:"tool_calls,omitempty"`
}

type ChatCompletionRequest struct {
	MaxCompletionTokens *int                   `json:"max_completion_tokens,omitempty"`
	ResponseFormat      map[string]interface{} `json:"response_format,omitempty"`
	StreamOptions       *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
	ParallelToolCalls *bool         `json:"parallel_tool_calls,omitempty"`
	Model             string        `json:"model"`
	Messages          []ChatMessage `json:"messages"`
	Tools             []ChatTool    `json:"tools,omitempty"`
	ToolChoice        interface{}   `json:"tool_choice,omitempty"`
	Temperature       *float64      `json:"temperature,omitempty"`
	TopP              *float64      `json:"top_p,omitempty"`
	MaxTokens         *int          `json:"max_tokens,omitempty"`
	Stream            bool          `json:"stream,omitempty"`
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
	Refusal   *string        `json:"refusal,omitempty"`
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
	Usage   *ChatUsage        `json:"usage,omitempty"`
	ID      string            `json:"id"`
	Object  string            `json:"object"` // "chat.completion.chunk"
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ChatChunkChoice `json:"choices"`
}

// ---------- 发给上游的 Responses API 请求 ----------

// Responses 将函数定义平铺；嵌入结构复用 Chat 的字段及 JSON 标签。
type ResponsesTool struct {
	Type string `json:"type"`
	ChatFunctionDef
}

// ResponsesInputItem 覆盖 message / function_call / function_call_output 三种输入项
type ResponsesInputItem struct {
	Type      string      `json:"type,omitempty"` // message 项可省略,仅留 role
	Role      string      `json:"role,omitempty"`
	Content   interface{} `json:"content,omitempty"`
	CallID    string      `json:"call_id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Arguments string      `json:"arguments,omitempty"`
	Output    *string     `json:"output,omitempty"`
}

type ResponsesRequest struct {
	Text              map[string]interface{} `json:"text,omitempty"`
	ParallelToolCalls *bool                  `json:"parallel_tool_calls,omitempty"`
	Model             string                 `json:"model"`
	Instructions      string                 `json:"instructions,omitempty"`
	Input             []ResponsesInputItem   `json:"input"`
	Tools             []ResponsesTool        `json:"tools,omitempty"`
	ToolChoice        interface{}            `json:"tool_choice,omitempty"`
	Temperature       *float64               `json:"temperature,omitempty"`
	TopP              *float64               `json:"top_p,omitempty"`
	MaxOutputTokens   *int                   `json:"max_output_tokens,omitempty"`
	Stream            bool                   `json:"stream,omitempty"`
}

// ---------- 上游 Responses API 的非流式响应 ----------

type ResponsesContentPart struct {
	Refusal string `json:"refusal,omitempty"`
	Type    string `json:"type"`
	Text    string `json:"text,omitempty"`
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
	Status string `json:"status"`
	Error  *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
	ID     string                `json:"id"`
	Object string                `json:"object"`
	Model  string                `json:"model"`
	Output []ResponsesOutputItem `json:"output"`
	Usage  *ResponsesUsage       `json:"usage,omitempty"`
}

// ---------- 上游 Responses API 的流式事件 ----------

type ResponsesStreamEvent struct {
	Type     string                `json:"type"`
	ItemID   string                `json:"item_id"`
	Delta    string                `json:"delta"`
	Item     *ResponsesOutputItem  `json:"item"`
	Response *ResponsesAPIResponse `json:"response"`
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

// convertUserContent: 将 Chat Completions 多模态数组映射为 Responses 输入格式
// text -> input_text, image_url -> input_image, 其余原样透传
func convertUserContent(c interface{}) interface{} {
	arr, ok := c.([]interface{})
	if !ok {
		return c // 字符串等,原样透传
	}
	out := make([]interface{}, 0, len(arr))
	for _, item := range arr {
		m, ok := item.(map[string]interface{})
		if !ok {
			out = append(out, item)
			continue
		}
		switch t, _ := m["type"].(string); t {
		case "text":
			m = maps.Clone(m)
			m["type"] = "input_text"
		case "image_url":
			m = maps.Clone(m)
			m["type"] = "input_image"
			// Chat 的 image_url 是对象；Responses 要求 URL 字符串和同级 detail。
			if image, ok := m["image_url"].(map[string]interface{}); ok {
				m["image_url"] = image["url"]
				if detail, exists := image["detail"]; exists {
					m["detail"] = detail
				}
			}
		}
		out = append(out, m)
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
				Output: ptr(contentToString(msg.Content)),
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
			tools = append(tools, ResponsesTool{Type: "function", ChatFunctionDef: *t.Function})
		}
	}

	// 新参数优先；指针保留“未传”和显式零值的区别。
	maxTokens := chatReq.MaxCompletionTokens
	if maxTokens == nil {
		maxTokens = chatReq.MaxTokens
	}
	// JSON Schema 在 Chat 中多包了一层 json_schema，Responses 使用 text.format。
	var text map[string]interface{}
	if chatReq.ResponseFormat != nil {
		format := maps.Clone(chatReq.ResponseFormat)
		if format["type"] == "json_schema" {
			if schema, ok := format["json_schema"].(map[string]interface{}); ok {
				format = maps.Clone(schema)
				format["type"] = "json_schema"
			}
		}
		text = map[string]interface{}{"format": format}
	}
	return &ResponsesRequest{
		Text:              text,
		ParallelToolCalls: chatReq.ParallelToolCalls,
		Model:             chatReq.Model,
		Instructions:      strings.Join(systemParts, "\n"),
		Input:             input,
		Tools:             tools,
		ToolChoice:        convertToolChoice(chatReq.ToolChoice),
		Temperature:       chatReq.Temperature,
		TopP:              chatReq.TopP,
		MaxOutputTokens:   maxTokens,
		Stream:            chatReq.Stream,
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
	proxyAPIKey := os.Getenv("PROXY_API_KEY")
	return func(w http.ResponseWriter, r *http.Request) {
		// 简单 CORS 支持
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if proxyAPIKey != "" && subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+proxyAPIKey)) != 1 {
			writeAPIError(w, "Invalid proxy API key", http.StatusUnauthorized)
			return
		}

		// 1. 读取并解析本地发送的 Chat Completions 请求
		var chatReq ChatCompletionRequest
		r.Body = http.MaxBytesReader(w, r.Body, 32<<20)
		decoder := json.NewDecoder(r.Body)
		// 不支持的参数直接报错，避免客户端误以为配置已生效。
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&chatReq); err != nil {
			writeAPIError(w, fmt.Sprintf("Invalid or unsupported Chat Completions payload: %v", err), http.StatusBadRequest)
			return
		}
		if err := decoder.Decode(new(interface{})); err != io.EOF {
			writeAPIError(w, "Expected one JSON request", http.StatusBadRequest)
			return
		}
		if chatReq.Model == "" || len(chatReq.Messages) == 0 {
			writeAPIError(w, "model and messages are required", http.StatusBadRequest)
			return
		}

		// 2. 转换为 Responses API 请求结构
		respReq := convertChatToResponsesReq(&chatReq)

		// 3. 构建发往上游 /v1/responses 的请求
		respPayload, err := json.Marshal(respReq)
		if err != nil {
			writeAPIError(w, err.Error(), http.StatusBadRequest)
			return
		}
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
		} else if auth := r.Header.Get("Authorization"); auth != "" && proxyAPIKey == "" {
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
			for _, key := range []string{"Content-Type", "Retry-After", "X-Request-Id"} {
				if value := resp.Header.Get(key); value != "" {
					w.Header().Set(key, value)
				}
			}
			w.WriteHeader(resp.StatusCode)
			io.Copy(w, resp.Body)
			return
		}

		// 4. 按流式/非流式分别转换响应
		if chatReq.Stream {
			handleStreamResponse(w, resp.Body, chatReq.Model, chatReq.StreamOptions != nil && chatReq.StreamOptions.IncludeUsage)
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
	finishReason, err := responseFinishReason(&out)
	if err != nil {
		writeAPIError(w, err.Error(), http.StatusBadGateway)
		return
	}
	var refusal strings.Builder
	for _, item := range out.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "refusal" {
					refusal.WriteString(part.Refusal)
				}
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

	var content interface{}
	if len(textParts) > 0 {
		content = strings.Join(textParts, "")
	} // 纯工具调用时保持 nil -> "content": null

	id := out.ID
	if id == "" {
		id = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
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
					Refusal:   refusal.String(),
					Role:      "assistant",
					Content:   content,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: convertUsage(out.Usage),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(chatResp)
}

// ==================== 6. 流式响应转换: Responses SSE -> Chat Completions SSE ====================

// ptr 让 index=0、output="" 等有效零值不会被 omitempty 丢弃。
func ptr[T any](v T) *T { return &v }

func writeAPIError(w http.ResponseWriter, message string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{"error": map[string]string{"message": message, "type": "proxy_error"}})
}

// responseFinishReason 统一两种响应的结束语义；截断和失败不能当作正常 stop。
func responseFinishReason(out *ResponsesAPIResponse) (string, error) {
	if out.Error != nil {
		return "", fmt.Errorf("upstream error: %s", out.Error.Message)
	}
	switch out.Status {
	case "", "completed":
		for _, item := range out.Output {
			if item.Type == "function_call" {
				return "tool_calls", nil
			}
		}
		return "stop", nil
	case "incomplete":
		if out.IncompleteDetails != nil {
			switch out.IncompleteDetails.Reason {
			case "max_output_tokens":
				return "length", nil
			case "content_filter":
				return "content_filter", nil
			}
		}
	}
	return "", fmt.Errorf("upstream response did not complete: %s", out.Status)
}

// convertUsage 兼容上游未提供 total_tokens 的情况。
func convertUsage(usage *ResponsesUsage) *ChatUsage {
	if usage == nil {
		return nil
	}
	total := usage.TotalTokens
	if total == 0 {
		total = usage.InputTokens + usage.OutputTokens
	}
	return &ChatUsage{PromptTokens: usage.InputTokens, CompletionTokens: usage.OutputTokens, TotalTokens: total}
}

func handleStreamResponse(w http.ResponseWriter, upstreamBody io.Reader, reqModel string, includeUsage bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAPIError(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	// 同一条流共享元数据，只更新 choices 和最终 usage。
	response := ChatCompletionChunk{
		ID:     fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: reqModel,
	}
	send := func(value interface{}) error {
		b, err := json.Marshal(value)
		if err != nil {
			return err
		}
		if _, err = fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	chunk := func(delta ChatChunkDelta, finish *string) error {
		response.Choices = []ChatChunkChoice{{Index: 0, Delta: delta, FinishReason: finish}}
		return send(response)
	}
	done := func() {
		if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err == nil {
			flusher.Flush()
		}
	}
	// 已开始 SSE 后只能通过事件报告错误；返回 true 表示停止读取上游。
	fail := func(err error) bool {
		if send(map[string]interface{}{"error": map[string]string{"message": err.Error(), "type": "upstream_error"}}) == nil {
			done()
		}
		return true
	}
	sentRole := false
	role := func() error {
		if sentRole {
			return nil
		}
		sentRole = true
		return chunk(ChatChunkDelta{Role: "assistant"}, nil)
	}
	// output_index 包含文本等项目；工具下标需按 item_id 单独连续编号。
	toolIndexByID := map[string]int{}
	// process 返回 true 表示已结束或客户端写入失败；调用方立即关闭上游响应。
	process := func(data string) bool {
		if data == "[DONE]" {
			return fail(fmt.Errorf("upstream ended before a terminal response event"))
		}
		var ev ResponsesStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return fail(fmt.Errorf("invalid upstream SSE JSON: %w", err))
		}
		switch ev.Type {
		case "response.output_item.added":
			if ev.Item == nil {
				return false
			}
			if ev.Item.Type == "message" {
				return role() != nil
			}
			if ev.Item.Type == "function_call" {
				if err := role(); err != nil {
					return true
				}
				if ev.Item.ID == "" {
					return fail(fmt.Errorf("tool call is missing item ID"))
				}
				if _, exists := toolIndexByID[ev.Item.ID]; exists {
					return false
				}
				idx := len(toolIndexByID)
				toolIndexByID[ev.Item.ID] = idx
				return chunk(ChatChunkDelta{ToolCalls: []ChatToolCall{{Index: ptr(idx), ID: ev.Item.CallID, Type: "function",
					Function: &ChatToolCallFunction{Name: ev.Item.Name, Arguments: ev.Item.Arguments}}}}, nil) != nil
			}
		case "response.output_text.delta", "response.refusal.delta":
			if err := role(); err != nil {
				return true
			}
			delta := ChatChunkDelta{Content: &ev.Delta}
			if ev.Type == "response.refusal.delta" {
				delta = ChatChunkDelta{Refusal: &ev.Delta}
			}
			return chunk(delta, nil) != nil
		case "response.function_call_arguments.delta":
			idx, exists := toolIndexByID[ev.ItemID]
			if !exists {
				return fail(fmt.Errorf("arguments received for unknown tool item %q", ev.ItemID))
			}
			return chunk(ChatChunkDelta{ToolCalls: []ChatToolCall{{Index: ptr(idx), Function: &ChatToolCallFunction{Arguments: ev.Delta}}}}, nil) != nil
		case "response.completed", "response.incomplete":
			if ev.Response == nil {
				return fail(fmt.Errorf("terminal event is missing response"))
			}
			if ev.Type == "response.incomplete" {
				ev.Response.Status = "incomplete"
			}
			reason, err := responseFinishReason(ev.Response)
			if err != nil {
				return fail(err)
			}
			if reason == "stop" && len(toolIndexByID) > 0 {
				reason = "tool_calls"
			}
			if role() != nil || chunk(ChatChunkDelta{}, &reason) != nil {
				return true
			}
			if includeUsage {
				// usage 单独发一帧，choices 必须为空数组。
				response.Choices = []ChatChunkChoice{}
				response.Usage = convertUsage(ev.Response.Usage)
				if err := send(response); err != nil {
					return true
				}
			}
			done()
			return true
		case "response.failed", "error":
			if ev.Response != nil && ev.Response.Error != nil {
				fail(fmt.Errorf("upstream error: %s", ev.Response.Error.Message))
			} else {
				fail(fmt.Errorf("upstream error: %s", data))
			}
			return true
		}
		return false
	}
	// SSE 以空行分帧，多条 data 行合并后才解析 JSON，忽略注释和其他字段。
	scanner := bufio.NewScanner(upstreamBody)
	scanner.Buffer(make([]byte, 64*1024), 8*1024*1024)
	var dataLines []string
	eventSize := 0
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if len(dataLines) > 0 && process(strings.Join(dataLines, "\n")) {
				return
			}
			dataLines = nil
			eventSize = 0
		} else if strings.HasPrefix(line, "data:") {
			data := strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " ")
			eventSize += len(data) + 1
			if eventSize > 8*1024*1024 {
				fail(fmt.Errorf("upstream SSE event exceeds 8 MiB"))
				return
			}
			dataLines = append(dataLines, data)
		}
	}
	if err := scanner.Err(); err != nil {
		fail(fmt.Errorf("reading upstream SSE: %w", err))
		return
	}
	if len(dataLines) > 0 && process(strings.Join(dataLines, "\n")) {
		return
	}
	fail(fmt.Errorf("upstream stream closed before a terminal response event"))
}

// ==================== 7. 主函数 ====================

func main() {
	// 上游只支持 Responses API 的服务器地址(通过环境变量 UPSTREAM_BASE_URL 配置)
	// 兼容 http://host:9695 与 http://host:9695/v1 两种写法
	upstreamBaseURL := envOr("UPSTREAM_BASE_URL", "http://172.20.156.12:9695/v1")
	responsesURL := joinResponsesURL(upstreamBaseURL)
	upstreamAPIKey := os.Getenv("UPSTREAM_API_KEY") // 可选:代理统一注入密钥
	port := envOr("PORT", "8081")
	port = strings.TrimPrefix(port, ":")
	// 默认只供本机使用；入口密钥与上游密钥分开，避免误转发。
	addr := net.JoinHostPort(envOr("HOST", "127.0.0.1"), port)
	if os.Getenv("PROXY_API_KEY") != "" && upstreamAPIKey == "" {
		log.Fatal("PROXY_API_KEY requires UPSTREAM_API_KEY")
	}

	proxy := handleChatToResponsesProxy(responsesURL, upstreamAPIKey)

	// 同时注册带/不带 /v1 前缀的端点,兼容各类客户端
	http.HandleFunc("/v1/chat/completions", proxy)
	http.HandleFunc("/chat/completions", proxy)

	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})

	log.Printf("代理已启动!本地 Chat Completions 地址: http://%s/v1/chat/completions", addr)
	log.Printf("请求将被自动转换为上游 Responses API: %s", responsesURL)
	log.Printf("提示: 可通过环境变量 UPSTREAM_BASE_URL / HOST / PORT 配置地址，PROXY_API_KEY 配置入口鉴权")

	server := &http.Server{Addr: addr, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 90 * time.Second}
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("服务启动失败: %v", err)
	}
}
