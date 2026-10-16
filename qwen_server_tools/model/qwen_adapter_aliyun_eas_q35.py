# -*- coding: utf-8 -*-
"""
Aliyun EAS / local Qwen adapter for VQA generation.

What this version does
----------------------
1) backend="aliyun_eas" never imports transformers.
2) For Aliyun OpenAI-compatible multimodal services, it sends image content as:
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
3) It keeps a fallback to legacy /infer_forward for older MLLM deployments.
4) It exposes the same surface your existing converter expects:
   - chat
   - generate
   - generate_on_image
   - generate_on_images
   - print_resource_report / save_resource_report
   - set_generation_defaults

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import base64
import io
import json
import time
import collections

import requests


def _strip_multiview_camera_header_text(text: str) -> str:
    """
    Remove the multi-view camera-layout prose and image placeholders from
    follow-up text turns. The first multimodal turn keeps this header so images
    remain aligned with camera labels; later turns should be text-only.
    """
    if not isinstance(text, str) or not text:
        return text

    s = text.replace("<IMAGE>", "<image>")
    end_markers = (
        "下文提及的前方/左前方/右前方/后方/左后方/右后方，均以此摄像头布局为准。",
        "下文提及的前方/左前方/右前方/左后方/右后方，均以此摄像头布局为准。",
        "When I mention FRONT / FRONT-LEFT / FRONT-RIGHT / BACK / BACK-LEFT / BACK-RIGHT below, they all refer to this fixed camera layout.",
        "When I mention FRONT / FRONT-LEFT / FRONT-RIGHT / BACK-LEFT / BACK-RIGHT below, they all refer to this fixed camera layout.",
        "When I mention FRONT / BACK / LEFT / RIGHT below, they all refer to this fixed camera layout.",
    )
    for marker in end_markers:
        idx = s.find(marker)
        if idx >= 0:
            s = s[idx + len(marker):]
            break

    return s.replace("<image>", "").strip()


def _human_bytes(n: Optional[int]) -> str:
    """
    将字节数转换为易读的人类友好格式（自动适配 B/KB/MB/GB/TB 单位）
    调用示例：
    _human_bytes(1023)       # 输出: "1023.00 B"
    _human_bytes(1024)       # 输出: "1.00 KB"
    _human_bytes(1536)       # 输出: "1.50 KB"
    _human_bytes(1048576)    # 输出: "1.00 MB"
    _human_bytes(None)       # 输出: "n/a"
    输出：成功返回带两位小数+单位的字符串，输入为 None 时返回 "n/a"
    """
    if n is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def _normalize_endpoint(endpoint: str) -> str:
    """
    标准化接口地址：移除字符串末尾的所有斜杠 /
    调用示例：
    _normalize_endpoint("https://api.test.com/")  # 输出: "https://api.test.com"
    _normalize_endpoint("/user/data//")          # 输出: "/user/data"
    _normalize_endpoint("login")                 # 输出: "login"
    输出：返回去除末尾所有斜杠后的字符串
    """
    return str(endpoint).rstrip("/")


def _pil_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    """
    将PIL图片对象转换为可直接嵌入网页的Data URI(base64)格式字符串
    调用示例：
    img = Image.open("test.jpg")
    _pil_to_data_uri(img)                  # 输出: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/..."
    _pil_to_data_uri(img, fmt="PNG")       # 输出: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    输出：返回图片的base64 Data URI字符串
    """
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """
    将PIL图片对象转换为纯Base64编码字符串
    调用示例：
    img = Image.open("test.jpg")
    _pil_to_base64(img)                  # 输出: "/9j/4AAQSkZJRgABAQEAAAAAAAD/..."
    _pil_to_base64(img, fmt="PNG")       # 输出: "iVBORw0KGgoAAAANSUhEUgAA..."
    输出：返回图片二进制数据的纯Base64编码字符串
    """
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class _EASResponse:
    """
    封装 EAS 服务响应结果的数据类，统一存储响应文本、路由和原始响应
    """
    text: str
    route: str
    raw: Any


class _AliyunEASBackend:
    """
    阿里云 EAS 模型服务后端客户端，用于初始化连接配置并发送推理请求
    调用示例：
    backend = _AliyunEASBackend(
        endpoint="https://eas-service.cn-hangzhou.aliyuncs.com/",
        token="your-auth-token",
        model_name="qwen2-7b"
    )
    输出：初始化完成，返回 _AliyunEASBackend 实例对象
    """
    def __init__(
        self,
        endpoint: str,          # 阿里云 EAS 服务访问地址
        token: str,             # EAS 服务鉴权令牌
        model_name: Optional[str] = None,       # 模型名称（可选）
        timeout: int = 300,           # 请求超时时间（秒）
        verify_ssl: bool = True,      # 是否校验 SSL 证书
        prefer_route: str = "auto",   # auto | chat_completions | infer_forward # 优先使用的接口路由类型
        verbose: bool = False,        # 是否开启调试日志
    ):
        self.endpoint = _normalize_endpoint(endpoint)
        self.token = token
        self.model_name = model_name
        self.timeout = int(timeout)
        self.verify_ssl = bool(verify_ssl)
        self.prefer_route = str(prefer_route or "auto")
        self.verbose = bool(verbose)
        self.session = requests.Session()           # 创建 requests 会话，复用连接提升性能
        self.last_route_used: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        """
        生成阿里云 EAS 服务请求所需的请求头字典
        调用示例：
        self._headers()
        输出：{"Authorization": "your-token", "Content-Type": "application/json"}
        """
        return {
            "Authorization": self.token,
            "Content-Type": "application/json",
        }

    def _messages_to_openai_content(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
    ) -> List[Dict[str, Any]]:
        """
        将内部消息格式转换为 OpenAI 兼容的消息体（支持文本+多模态图片）
        调用示例：
        messages = [{"role": "user", "content": "这张图是什么"}]
        images = [Image.open("test.jpg")]
        _messages_to_openai_content(messages, images)
        输出：[{"role": "user", "content": [{"type": "image_url", ...}, {"type": "text", ...}]}]
        输出：返回符合 OpenAI API 规范的 messages 结构
        """
        out: List[Dict[str, Any]] = []
        used_images = False
        user_turn_count = 0
        user_total = sum(1 for m in messages if str(m.get("role", "user")) == "user")
        deferred_camera_header = ""
        if images and user_total > 1:
            header_end_markers = (
                "下文提及的前方/左前方/右前方/后方/左后方/右后方，均以此摄像头布局为准。",
                "下文提及的前方/左前方/右前方/左后方/右后方，均以此摄像头布局为准。",
                "When I mention FRONT / FRONT-LEFT / FRONT-RIGHT / BACK / BACK-LEFT / BACK-RIGHT below, they all refer to this fixed camera layout.",
                "When I mention FRONT / FRONT-LEFT / FRONT-RIGHT / BACK-LEFT / BACK-RIGHT below, they all refer to this fixed camera layout.",
                "When I mention FRONT / BACK / LEFT / RIGHT below, they all refer to this fixed camera layout.",
            )
            for m in reversed(messages):
                if str(m.get("role", "user")) != "user":
                    continue
                c = m.get("content", "")
                if isinstance(c, list):
                    continue
                s = str(c).replace("<IMAGE>", "<image>")
                for marker in header_end_markers:
                    idx = s.find(marker)
                    if idx >= 0:
                        deferred_camera_header = s[:idx + len(marker)].strip() + "\n\n"
                        break
                if deferred_camera_header:
                    break
        
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            is_user = str(role) == "user"
            is_first_user_turn = is_user and user_turn_count == 0
            if is_user:
                user_turn_count += 1

             # ====================== 情况1：content 本身是列表（多段内容）======================
            if isinstance(content, list):
                segs = []
                for seg in content:
                    if not isinstance(seg, dict):
                        segs.append({"type": "text", "text": str(seg)})
                        continue
                    seg_type = seg.get("type", "text")
                    if seg_type in {"image", "image_url"}:
                        if is_user and not is_first_user_turn:
                            continue
                        img_obj = seg.get("image")
                        if isinstance(img_obj, Image.Image):
                            segs.append({
                                "type": "image_url",
                                "image_url": {"url": _pil_to_data_uri(img_obj)},
                            })
                        elif isinstance(seg.get("image_url"), dict) and "url" in seg["image_url"]:
                            segs.append({
                                "type": "image_url",
                                "image_url": {"url": seg["image_url"]["url"]},
                            })
                        elif isinstance(img_obj, str) and img_obj:
                            segs.append({
                                "type": "image_url",
                                "image_url": {"url": img_obj},
                            })
                    elif seg_type == "text":
                        text = str(seg.get("text", ""))
                        if is_user and not is_first_user_turn:
                            text = _strip_multiview_camera_header_text(text)
                        segs.append({"type": "text", "text": text})
                out.append({"role": role, "content": segs})
                continue
            # ====================== 情况2：content 是普通文本 ======================
            # 如果是用户消息 + 有传入图片 + 还没使用过图片 → 自动把图片拼接到内容里
            if (not used_images) and images and is_user and is_first_user_turn:
                content_str = str(content)
                if deferred_camera_header and "六张同步图像" not in content_str and "six synchronized images" not in content_str:
                    content_str = deferred_camera_header + content_str
                # 如果文字中包含 <image> 标记，则在标记位置交织插入图片（图文紧邻）
                normalized = content_str.replace('<IMAGE>', '<image>')
                if '<image>' in normalized:
                    parts = normalized.split('<image>')
                    segs = []
                    img_idx = 0
                    for i, part in enumerate(parts):
                        if part.strip():
                            segs.append({"type": "text", "text": part})
                        if i < len(parts) - 1 and img_idx < len(images):
                            img = images[img_idx]
                            if img is not None:
                                segs.append({"type": "image_url", "image_url": {"url": _pil_to_data_uri(img)}})
                            img_idx += 1
                    # 剩余未消费的图片追加到末尾（兜底）
                    while img_idx < len(images):
                        img = images[img_idx]
                        if img is not None:
                            segs.append({"type": "image_url", "image_url": {"url": _pil_to_data_uri(img)}})
                        img_idx += 1
                    out.append({"role": role, "content": segs})
                else:
                    # 无 <image> 标记：保持原有行为，所有图片在前、文字在后
                    segs = [
                        {"type": "image_url", "image_url": {"url": _pil_to_data_uri(img)}}
                        for img in images if img is not None
                    ]
                    if content_str.strip():
                        segs.append({"type": "text", "text": content_str})
                    out.append({"role": role, "content": segs})
                used_images = True
            else:
                cleaned = str(content)
                if is_user and not is_first_user_turn:
                    cleaned = _strip_multiview_camera_header_text(cleaned)
                else:
                    cleaned = cleaned.replace('<image>', '').replace('<IMAGE>', '')
                out.append({"role": role, "content": cleaned})
        return out

    def _messages_to_prompt_and_history(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, List[List[str]]]:
        """
        将多轮对话消息拆分为：当前提问(prompt) + 历史对话(history)，并提取system指令
        调用示例：
        messages = [
            {"role": "system", "content": "你是AI助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "今天天气"}
        ]
        _messages_to_prompt_and_history(messages)
        输出：("你是AI助手\n\n今天天气", [["你好", "你好！"]])
        输出：返回 (拼接system的最终prompt, 多轮对话历史列表[[用户问题, 助手回复], ...])
        """
        system_prefix: List[str] = []
        linear: List[Tuple[str, str]] = []
        user_turn_count = 0
       
        # 遍历所有消息，分离 system 指令与对话内容
        for m in messages:
            role = str(m.get("role", "user"))
            content = m.get("content", "")

            # 如果内容是列表格式（多模态/分段内容），仅提取纯文本
            if isinstance(content, list):
                texts = []
                for seg in content:
                    if isinstance(seg, dict) and seg.get("type") == "text":
                        texts.append(str(seg.get("text", "")))
                content = "\n".join([t for t in texts if t]) # 合并提取到的所有文本
            content = str(content)
            if role == "user":
                if user_turn_count > 0:
                    content = _strip_multiview_camera_header_text(content)
                user_turn_count += 1

            # 收集 system 角色的提示词
            if role == "system":
                if content.strip():
                    system_prefix.append(content.strip())
            else:
                linear.append((role, content))  # 收集 user/assistant 对话

        # 最终返回的当前提问 & 多轮对话历史
        prompt = ""
        history: List[List[str]] = []
        last_user_idx = None

        # 倒序查找最后一条 user 消息（当前要发送的问题）
        for i in range(len(linear) - 1, -1, -1):
            if linear[i][0] == "user":
                last_user_idx = i
                break
        # 如果没有找到 user 消息，直接返回 system 指令 + 空历史
        if last_user_idx is None:
            prompt = "\n".join(system_prefix).strip()
            return prompt, history
        # 按 [user, assistant] 配对，构建多轮历史对话
        for i in range(0, last_user_idx, 2):
            if i + 1 < last_user_idx and linear[i][0] == "user" and linear[i + 1][0] == "assistant":
                history.append([linear[i][1], linear[i + 1][1]])

        prompt = linear[last_user_idx][1] # 最后一条 user 消息作为当前 prompt
        # 将 system 指令拼接在 prompt 最前方
        if system_prefix:
            prefix = "\n\n".join(system_prefix).strip()
            prompt = prefix + "\n\n" + prompt if prompt.strip() else prefix

        return prompt, history

    def _post_chat_completions(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        do_sample: bool,
    ) -> _EASResponse:
        """
        调用阿里云 EAS 服务 /v1/chat/completions 接口发送对话推理请求（支持文本+多模态）
        调用示例：
        messages = [{"role": "user", "content": "图片里是什么"}]
        images = [Image.open("test.jpg")]
        _post_chat_completions(messages, images, 1024, 0.7, 0.95, True)
        输出：_EASResponse(text='图片里是...', route='chat_completions', raw={...})
        输出：返回封装后的 EAS 响应对象
        """
        url = f"{self.endpoint}/v1/chat/completions"
        
        payload: Dict[str, Any] = {
            "messages": self._messages_to_openai_content(messages, images=images),
            "stream": False,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_new_tokens),
            "chat_template_kwargs": {"enable_thinking": False},
        }
         
        if self.model_name:# 可选：传入模型名称
            payload["model"] = self.model_name

        # 扩展参数：关闭模型思考过程，仅输出结果
        # payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        # 发送 POST 请求到 EAS 服务
        # ====================== 计时：单帧 EAS 耗时 ======================
        # start_frame = time.perf_counter()
        resp = self.session.post(
            url,
            json=payload,
            headers=self._headers(),        # 携带鉴权请求头
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        # 计算耗时
        # end_frame = time.perf_counter()
        # duration = end_frame - start_frame
        # # logger.info(f"[✅ 单次询问 EAS 耗时] frame={idx} | {duration:.2f} 秒 ({duration*1000:.0f} ms)")
        # logger.info(f"[✅ EAS 请求耗时] {duration:.2f} 秒 ({duration*1000:.0f} ms)")

        # HTTP 请求失败时抛出异常
        if resp.status_code >= 400:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {url}\n{resp.text[:2000]}",
                response=resp,
            )

        # 解析接口返回的 JSON 数据
        data = resp.json()
        text = ""

        # 兼容多种返回格式，提取模型生成文本
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            try:
                text = data["choices"][0]["text"]
            except Exception:
                text = str(data)
        return _EASResponse(text=str(text).strip(), route="chat_completions", raw=data)

    def _post_infer_forward(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        do_sample: bool,
    ) -> _EASResponse:
        """
        调用阿里云 EAS 服务 /infer_forward 原生接口发送推理请求（兼容传统大模型格式）
        调用示例：
        messages = [{"role": "user", "content": "图片里有什么"}]
        images = [Image.open("test.jpg")]
        _post_infer_forward(messages, images, 1024, 0.7, 0.95, True)
        输出：_EASResponse(text='图片里是...', route='infer_forward', raw={...})
        输出：返回封装后的 EAS 响应对象
        """
        url = f"{self.endpoint}/infer_forward"  # 拼接 EAS 服务原生推理接口地址
        
        prompt, history = self._messages_to_prompt_and_history(messages) # 将消息格式转换为模型需要的 prompt + 对话历史
        # 处理图片：仅取第一张有效图片并转为纯 base64 格式
        image_b64 = None
        user_count = sum(1 for m in messages if str(m.get("role", "user")) == "user")
        if user_count <= 1 and images:
            valid_images = [img for img in images if img is not None]
            if valid_images:
                image_b64 = _pil_to_base64(valid_images[0])

        # 构造 /infer_forward 接口请求体
        payload: Dict[str, Any] = {
            "prompt": prompt,                       # 当前提问（含system指令）
            "image": image_b64,                     # 图片base64编码
            "chat_history": history,                # 多轮对话历史
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_new_tokens),
            "use_stream": False,                    # 关闭流式输出
        }

        # 发送 POST 请求到 EAS 模型服务
        resp = self.session.post(
            url,
            json=payload,
            headers=self._headers(),        # 携带鉴权请求头
            timeout=self.timeout,
            verify=self.verify_ssl,
        )

        # HTTP 请求异常时抛出错误
        if resp.status_code >= 400:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {url}\n{resp.text[:2000]}",
                response=resp,
            )

        # 灵活解析返回结果，兼容多种返回格式
        text_raw = resp.text
        text = text_raw
        try:
            js = resp.json()
            if isinstance(js, dict):
                if "response" in js:
                    text = js["response"]
                elif "text" in js:
                    text = js["text"]
                else:
                    text = json.dumps(js, ensure_ascii=False)
            elif isinstance(js, str):
                text = js
            else:
                text = json.dumps(js, ensure_ascii=False)
            raw = js
        except Exception:
            raw = text_raw
        return _EASResponse(text=str(text).strip(), route="infer_forward", raw=raw)

    def _diagnose_failures(self, errors: List[Tuple[str, Exception]]) -> str:
        """
        生成阿里云 EAS 路由探测失败的诊断错误信息（格式化输出）
        调用示例：
        errors = [("chat_completions", ConnectionError()), ("infer_forward", TimeoutError())]
        _diagnose_failures(errors)
        输出：拼接好的完整错误诊断文本
        输出：返回人类可读的路由失败原因与诊断说明
        """
        lines = ["Aliyun EAS route probing failed."]
        for route, err in errors:
            lines.append(f"- {route}: {err}")
        lines.append(
            "Diagnosis: the current EAS endpoint either does not expose the attempted multimodal route, or it is not the expected VL service."
        )
        return "\n".join(lines)

    def chat(
        self,
        messages: List[Dict[str, Any]],             # 对话消息列表（user/assistant/system）
        images: Optional[List[Image.Image]] = None, # 多模态图片（可选）
        max_new_tokens: int = 256,          # 最大生成token数
        top_p: float = 0.9,                 # 核采样参数
        temperature: float = 0.0,           # 生成温度（0=最稳定）
        do_sample: bool = False,            # 是否随机采样
    ) -> str:
        """
        对外暴露的对话主方法：自动尝试可用路由，调用阿里云 EAS 多模态模型生成回复
        调用示例：
        messages = [{"role": "user", "content": "图片里是什么"}]
        images = [Image.open("test.jpg")]
        chat(messages, images)
        输出："图片里是一只猫。"
        输出：返回模型生成的文本回答
        """
        routes: List[str]
        if self.prefer_route == "chat_completions":
            routes = ["chat_completions"]
        elif self.prefer_route == "infer_forward":
            routes = ["infer_forward"]
        else:
            routes = ["chat_completions", "infer_forward"]

        errors: List[Tuple[str, Exception]] = []
        for route in routes:
            try:
                if route == "chat_completions":
                    out = self._post_chat_completions(messages, images, max_new_tokens, top_p, temperature, do_sample)
                else:
                    out = self._post_infer_forward(messages, images, max_new_tokens, top_p, temperature, do_sample)
                self.last_route_used = out.route
                return out.text
            except Exception as e:
                errors.append((route, e))
                continue
        raise RuntimeError(self._diagnose_failures(errors))

    def generate(
        self,
        text: str,          # 纯文本提问内容
        images: Optional[List[Image.Image]] = None,     # 多模态图片列表（可选）
        max_new_tokens: int = 256,      # 最大生成token数
        temperature: float = 0.0,       # 生成温度，控制随机性
        do_sample: bool = False,        # 是否随机采样
        top_p: float = 0.9,             # 核采样参数
    ) -> str:
        """
        简化版文本/多模态生成接口：直接传入文本+图片获取回答，封装chat方法
        调用示例：
        generate("图片里是什么", images=[Image.open("test.jpg")])
        输出："图片里是一只可爱的小狗"
        输出：返回模型生成的文本结果
        """
        return self.chat(
            messages=[{"role": "user", "content": text}],
            images=images,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
        )

    def generate_on_image(self, text: str, image: Image.Image, **kw) -> str:
        """
        简化单图多模态生成：传入单张图片 + 文本提问，直接返回回答
        调用示例：
        generate_on_image("这张图里有什么", Image.open("test.jpg"))
        输出："图里是一只猫"
        输出：返回模型生成的文本结果
        """
        return self.generate(text=text, images=[image], **kw)

    def generate_on_images(self, text: str, images: List[Image.Image], **kw) -> str:
        """
        多图多模态生成：传入多张图片 + 文本提问，直接返回回答
        调用示例：
        generate_on_images("对比两张图的区别", [Image.open("1.jpg"), Image.open("2.jpg")])
        输出："第一张图是...，第二张图是..."
        输出：返回模型生成的文本结果
        """
        return self.generate(text=text, images=images, **kw)

    def resource_report(self) -> Dict[str, Any]:
        """
        获取当前阿里云 EAS 客户端资源使用报告（状态信息）
        调用示例：
        resource_report()
        输出：{"timestamp":"2025-12-29 15:30:00","backend":"aliyun_eas",...}
        输出：返回包含服务信息的字典
        """
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "aliyun_eas",
            "endpoint": self.endpoint,
            "model_name": self.model_name,
            "last_route_used": self.last_route_used,
        }

    def print_resource_report(self) -> None:
        """
        打印美化版资源报告到控制台
        调用示例：
        print_resource_report()
        输出：格式化打印时间、后端、地址、模型、最后使用路由
        输出：无返回值，直接输出到终端
        """
        rep = self.resource_report()
        print("\n===== Aliyun EAS Adapter Report =====")
        for k, v in rep.items():
            print(f"{k}: {v}")
        print("====================================\n")

    def save_resource_report(self, path: str) -> None:
        """
        将资源报告保存到 JSON 文件
        调用示例：
        save_resource_report("report.json")
        输出：无返回值，文件写入成功
        """
        rep = self.resource_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)


class _LocalHFBackend:
    """
    本地 HuggingFace 多模态模型后端：加载本地 VLM 模型（图文问答），支持量化、加速、自定义配置
    调用示例：
    backend = _LocalHFBackend(
        model_dir="./qwen-vl-chat",
        device_map="auto",
        load_4bit=True
    )
    输出：完成本地模型加载，返回 _LocalHFBackend 实例对象
    输出：无直接返回，成功后可使用模型进行推理
    """
    def __init__(
        self,
        model_dir: str,         # 本地模型文件夹路径
        device_map: Union[str, Dict[str, int]] = "auto",    # 设备分配策略（auto/cuda:0等）
        *,
        torch_dtype=None,       # 精度类型（float16/bfloat16）
        min_pixels: Optional[int] = None,   # 图片最小像素
        max_pixels: Optional[int] = None,   # 图片最大像素
        load_4bit: bool = False,            # 是否开启4bit量化加载
        load_8bit: bool = False,            # 是否开启8bit量化加载
        llm_int8_enable_fp32_cpu_offload: bool = False, # 8bit时是否CPU卸载
        flash_attn: bool = False,           # 是否开启FlashAttention加速
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,    # 显存限制
    ):
        # 导入依赖库（延迟导入，避免未安装时报错）
        import torch
        # 优先使用图文转文本模型类，兼容新版transformers
        try:
            from transformers import AutoModelForImageTextToText as _AutoModelClass
        except Exception:
            # 兼容旧版视觉-文本模型类
            from transformers import AutoModelForVision2Seq as _AutoModelClass
        from transformers import AutoProcessor
        # 尝试导入量化配置
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            BitsAndBytesConfig = None

        self._torch = torch
        # 构造处理器初始化参数
        proc_kwargs: Dict[str, Any] = {"trust_remote_code": True, "use_fast": False}
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = int(max_pixels)

        # 加载模型处理器（处理文本+图片预处理）
        self.processor = AutoProcessor.from_pretrained(model_dir, **proc_kwargs)
        # 获取分词器（兼容processor内嵌/独立tokenizer）
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.processor
        self.tokenizer = tokenizer

        # 构造模型加载参数
        model_kwargs: Dict[str, Any] = dict(trust_remote_code=True, device_map=device_map)
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        
        # 4bit量化配置（低显存加载大模型）
        if load_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes is required to enable 4-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        # 8bit量化配置
        elif load_8bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes is required to enable 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=bool(llm_int8_enable_fp32_cpu_offload),
            )
        # 显存限制配置
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        # 开启FlashAttention加速
        if flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        # 加载模型，失败时自动移除不兼容的attention参数重试
        try:
            self.model = _AutoModelClass.from_pretrained(model_dir, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            self.model = _AutoModelClass.from_pretrained(model_dir, **model_kwargs)
        self.model.eval() # 设置模型为评估模式（禁用dropout，推理模式）

    @property
    def torch(self):
        """
        获取模型加载时使用的 torch 库对象（安全引用，避免重复导入）
        调用示例：
        backend = _LocalHFBackend(model_dir="./model")
        backend.torch  # 直接访问，无需加括号
        输出：返回已导入的 torch 模块对象
        """
        return self._torch

    def chat(
        self,
        messages: List[Dict[str, Any]],             # 对话消息列表（system/user/assistant）
        images: Optional[List[Image.Image]] = None, # 多模态图片列表
        max_new_tokens: int = 256,      # 最大生成token数
        top_p: float = 0.9,             # 核采样参数
        temperature: float = 0.0,       # 生成温度
        do_sample: bool = False,        # 是否随机采样生成
    ) -> str:
        """
        本地多模态模型核心对话方法：处理文本+图片对话，执行本地推理并返回回答
        调用示例：
        messages = [{"role": "user", "content": "图片里是什么"}]
        images = [Image.open("test.jpg")]
        chat(messages, images)
        输出示例："图片里是一只可爱的小猫"
        输出：返回模型生成的文本回答
        """
        torch = self._torch
        # 开启推理模式，禁用梯度计算，提升速度、节省显存
        with torch.inference_mode():
            # 过滤无效图片
            if images:
                images = [img for img in images if img is not None]
            mm_messages: List[Dict[str, Any]] = []
            used_images = False # 标记图片是否已加入消息

            # 遍历原始消息，构建模型支持的多模态消息格式
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                # 如果内容已是列表格式（多段文本/图片），直接保留
                if isinstance(content, list):
                    mm_messages.append(m)
                    continue
                # 用户消息 + 存在未使用图片 → 把图片+文本打包为多模态格式
                if (not used_images) and images and role == "user":
                    segments: List[Dict[str, Any]] = []
                    # 先添加所有图片
                    for img in images:
                        segments.append({"type": "image", "image": img})
                    # 再添加文本内容
                    if isinstance(content, str) and content.strip():
                        segments.append({"type": "text", "text": content})
                    mm_messages.append({"role": role, "content": segments})
                    used_images = True   # 标记图片已使用
                else:
                    # 普通消息统一转为纯文本分段格式
                    mm_messages.append({"role": role, "content": [{"type": "text", "text": str(content)}]})

            # 使用处理器生成模型所需的对话模板字符串
            chat_str = self.processor.apply_chat_template(mm_messages, tokenize=False, add_generation_prompt=True)
            # 预处理文本+图片，转为模型需要的张量格式，并放到模型所在设备
            inputs = self.processor(text=[chat_str], images=images if images else None, return_tensors="pt").to(self.model.device)
            # 构造生成参数
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=bool(do_sample), use_cache=True)
            # 采样模式下添加随机性参数
            if do_sample:
                gen_kwargs["top_p"] = float(top_p)
                gen_kwargs["temperature"] = float(temperature)
            out = self.model.generate(**inputs, **gen_kwargs)   # 执行模型推理生成
            gen = out[:, inputs["input_ids"].shape[-1]:]        # 截取生成部分（去掉输入的prompt部分）
            # 解码生成结果，去掉特殊token并返回
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def generate(self, text: str, images: Optional[List[Image.Image]] = None, max_new_tokens: int = 256,
                 temperature: float = 0.0, do_sample: bool = False, top_p: float = 0.9) -> str:
        return self.chat(
            messages=[{"role": "user", "content": text}],
            images=images,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
        )

    def generate_on_image(self, text: str, image: Image.Image, **kw) -> str:
        return self.generate(text=text, images=[image], **kw)

    def generate_on_images(self, text: str, images: List[Image.Image], **kw) -> str:
        return self.generate(text=text, images=images, **kw)

    def resource_report(self) -> Dict[str, Any]:
        '''
        接口一样，只是里面报告内容不同
        云端：报告模型服务地址、路由
        本地：报告GPU、显存、参数、量化
        '''
        torch = self._torch
        info = {"cuda_available": torch.cuda.is_available(), "gpus": []}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "id": i,
                    "name": prop.name,
                    "total": prop.total_memory,
                    "total_human": _human_bytes(prop.total_memory),
                    "allocated": torch.cuda.memory_allocated(i),
                    "allocated_human": _human_bytes(torch.cuda.memory_allocated(i)),
                    "reserved": torch.cuda.memory_reserved(i),
                    "reserved_human": _human_bytes(torch.cuda.memory_reserved(i)),
                })

        total = 0
        per_dtype = collections.Counter()
        per_device = collections.Counter()
        try:
            for p in self.model.parameters():
                n = p.numel()
                total += n
                per_dtype[str(p.dtype)] += n
                per_device[str(p.device)] += n
        except Exception:
            pass

        fp = None
        try:
            fp = int(self.model.get_memory_footprint())
        except Exception:
            fp = None

        dm = getattr(self.model, "hf_device_map", None)
        if isinstance(dm, dict):
            from collections import Counter
            summary = dict(Counter(map(str, dm.values())))
            dm_obj = {"map": {k: str(v) for k, v in dm.items()}, "summary": summary}
        else:
            dm_obj = {"map": None, "summary": {}}

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "local",
            "model_class": self.model.__class__.__name__,
            "processor_class": self.processor.__class__.__name__,
            "gpu": info,
            "params": {
                "total_params": int(total),
                "per_dtype_params": dict(per_dtype),
                "per_device_params": dict(per_device),
            },
            "memory_footprint": {"bytes": fp, "human": _human_bytes(fp)},
            "device_map": dm_obj,
            "quantization": {
                "is_loaded_in_4bit": bool(getattr(self.model, "is_loaded_in_4bit", False)),
                "is_loaded_in_8bit": bool(getattr(self.model, "is_loaded_in_8bit", False)),
            },
        }

    def print_resource_report(self) -> None:
        rep = self.resource_report()
        print("\n===== Local HF Qwen Resource Report =====")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("========================================\n")

    def save_resource_report(self, path: str) -> None:
        rep = self.resource_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)


class QwenVL:
    """
    QwenVL 多模态模型统一调用入口（本地/阿里云EAS云端一键切换）
    自动根据 backend 参数选择本地模型推理 或 阿里云EAS云端推理
    调用示例：
    # 本地模型
    model = QwenVL(model_dir="./qwen_vl_model", backend="local")
    # 阿里云EAS云端
    model = QwenVL(backend="aliyun_eas", eas_endpoint="https://xxx", eas_token="xxx")
    输出：初始化完成，返回可直接使用的 QwenVL 调用对象
    输出：统一接口，本地/云端用法完全一致
    """
    def __init__(
        self,
        model_dir: Optional[str] = None,        # 本地模型路径（仅local使用）
        device_map: Union[str, Dict[str, int]] = "auto",       # 设备分配策略（auto/cuda:0）
        *,
        backend: str = "local",             # 后端类型：local / aliyun_eas
        torch_dtype=None,                   # 模型精度（float16 / bfloat16）
        min_pixels: Optional[int] = None,   # 图片最小像素
        max_pixels: Optional[int] = None,   # 图片最大像素
        load_4bit: bool = False,            # 本地4bit量化加载
        load_8bit: bool = False,            # 本地8bit量化加载
        llm_int8_enable_fp32_cpu_offload: bool = False,     # 8bit量化CPU卸载
        flash_attn: bool = False,           # 开启FlashAttention加速
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,    # 显存限制
        eas_endpoint: Optional[str] = None,     # 阿里云EAS服务地址
        eas_token: Optional[str] = None,        # 阿里云EAS鉴权token
        eas_model_name: Optional[str] = None,   # EAS模型名称（可选）
        eas_timeout: int = 300,                 # EAS请求超时时间
        eas_verify_ssl: bool = True,            # EAS是否校验SSL证书
        eas_prefer_route: str = "auto",         # EAS优先路由：auto/chat_completions/infer_forward
        verbose: bool = False,                  # 调试日志开关
    ):
        self.backend = str(backend or "local").lower() # 统一后端类型（小写标准化）
        # 生成参数默认值（全局通用）
        self._generation_defaults = {
            "max_new_tokens": 256,
            "top_p": 0.9,
            "temperature": 0.0,
        }
        # ====================== 初始化阿里云EAS云端后端 ======================
        if self.backend == "aliyun_eas":
            # 云端必须传入 endpoint 和 token
            if not eas_endpoint or not eas_token:
                raise ValueError("backend='aliyun_eas' requires eas_endpoint and eas_token")
            # 初始化阿里云EAS客户端
            self._impl = _AliyunEASBackend(
                endpoint=eas_endpoint,
                token=eas_token,
                model_name=eas_model_name,
                timeout=eas_timeout,
                verify_ssl=eas_verify_ssl,
                prefer_route=eas_prefer_route,
                verbose=verbose,
            )
        # ====================== 初始化本地模型后端 ======================
        else:
            # 本地必须传入模型路径
            if not model_dir:
                raise ValueError("backend='local' requires model_dir")
            # 初始化本地HuggingFace模型
            self._impl = _LocalHFBackend(
                model_dir=model_dir,
                device_map=device_map,
                torch_dtype=torch_dtype,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                load_4bit=load_4bit,
                load_8bit=load_8bit,
                llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
                flash_attn=flash_attn,
                max_memory=max_memory,
            )

    def set_generation_defaults(self, *, max_new_tokens: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None) -> None:
        """
        设置全局默认的文本生成参数，后续调用无需重复传参
        调用示例：
        model.set_generation_defaults(max_new_tokens=1024, temperature=0.7)
        输出：无返回值，全局生成参数已更新
        输出：后续所有 chat/generate 自动使用新的默认值
        """
        # 如果传入了参数，则更新对应的全局默认值
        if max_new_tokens is not None:
            self._generation_defaults["max_new_tokens"] = int(max_new_tokens)
        if top_p is not None:
            self._generation_defaults["top_p"] = float(top_p)
        if temperature is not None:
            self._generation_defaults["temperature"] = float(temperature)

    def _resolve_gen(self, max_new_tokens: Optional[int], top_p: Optional[float], temperature: Optional[float]) -> Tuple[int, float, float]:
        """
        解析生成参数：优先使用传入参数，无传入则使用全局默认值
        调用示例：
        _resolve_gen(max_new_tokens=None, top_p=0.95, temperature=None)
        输出：(256, 0.95, 0.0)
        输出：返回最终确定的（最大token数, top_p, 温度）
        """
        return (
            int(self._generation_defaults["max_new_tokens"] if max_new_tokens is None else max_new_tokens),
            float(self._generation_defaults["top_p"] if top_p is None else top_p),
            float(self._generation_defaults["temperature"] if temperature is None else temperature),
        )

    def chat(self, messages: List[Dict[str, Any]], images: Optional[List[Image.Image]] = None,
             max_new_tokens: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None,
             do_sample: bool = False) -> str:
        mnt, tp, temp = self._resolve_gen(max_new_tokens, top_p, temperature)
        return self._impl.chat(messages=messages, images=images, max_new_tokens=mnt, top_p=tp, temperature=temp, do_sample=do_sample)

    def generate(self, text: str, images: Optional[List[Image.Image]] = None, max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, do_sample: bool = False, top_p: Optional[float] = None) -> str:
        mnt, tp, temp = self._resolve_gen(max_new_tokens, top_p, temperature)
        return self._impl.generate(text=text, images=images, max_new_tokens=mnt, temperature=temp, do_sample=do_sample, top_p=tp)

    def generate_on_image(self, text: str, image: Image.Image, **kw) -> str:
        return self.generate(text=text, images=[image], **kw)

    def generate_on_images(self, text: str, images: List[Image.Image], **kw) -> str:
        return self.generate(text=text, images=images, **kw)

    def resource_report(self) -> Dict[str, Any]:
        rep = self._impl.resource_report()
        rep["generation_defaults"] = dict(self._generation_defaults)
        return rep

    def print_resource_report(self) -> None:
        rep = self.resource_report()
        print("\n===== Qwen Adapter Report =====")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("===============================\n")

    def save_resource_report(self, path: str) -> None:
        rep = self.resource_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
