import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 配置日志
logging.basicConfig(
    filename='dataset_generation.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='ANSI'
)


def split_chapters(file_path: Path) -> List[Dict[str, str]]:
    """
    按章节切分小说文本，提取章节标题和内容

    参数:
        file_path: 小说全文TXT文件路径

    返回:
        章节列表，每个元素为字典{"title": 章节名, "content": 章节内容}
    """
    try:
        # 读取文件（处理BOM问题）
        with open(file_path, 'r', encoding='ANSI') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"读取文件失败: {str(e)}")
        raise RuntimeError(f"读取文件 {file_path} 失败: {str(e)}")

    # 优化章节匹配正则：支持更多标题格式（带卷名、副标题等）
    chapter_pattern = r'(第[零一二三四五六七八九十百千万\d]+章(?:\s*[^\n]+)?)\s*(.*?)(?=\n\s*第[零一二三四五六七八九十百千万\d]+章|$)'
    chapters = re.findall(chapter_pattern, content, re.DOTALL | re.IGNORECASE)

    chapter_list = []
    for title_part, content_part in chapters:
        # 清理标题（去除首尾空白和多余换行）
        title = re.sub(r'\s+', ' ', title_part).strip()
        # 清理内容（去除多余空白行）
        content_lines = [line.strip() for line in content_part.split('\n') if line.strip()]
        content = '\n'.join(content_lines)

        if title and (content or len(chapter_list) == 0):  # 允许第一章内容为空
            chapter_list.append({"title": title, "content": content})

    print(f"成功切分 {len(chapter_list)} 个章节")
    return chapter_list


def get_model_json_output(chapter: Dict[str, str], model: ChatOpenAI) -> Dict[str, Any]:
    """
    将单章内容传入模型，生成魏无羡对话样本的JSON数据

    参数:
        chapter: 包含"title"和"content"的章节字典
        model: 配置好的LangChain模型实例

    返回:
        包含章节标题和对话样本的字典
    """
    title = chapter["title"]
    content = chapter["content"]

    # 优化提示词：增加示例、明确场景要求、强调格式严谨性
    prompt = f"""
    请严格按以下四步处理《魔道祖师》章节《{title}》，**仅基于原文实际内容**提取魏无羡的对话，**不预设剧情阶段、性格或互动角色**，一切以带引号的直接对话及其上下文为唯一依据：

    ### 1. 阶段与性格：动态识别，禁止预设
    - **不强制锁定任何时期**（如“重生时期”），需根据本章实际情节判断魏无羡所处阶段（如少年时期、乱葬岗时期、重生时期等）；
    - **性格描述必须源自本章对话表现**（如调侃、愤怒、疲惫、疯癫、沉稳等），不得套用固定模板；
    - 所有判断必须有原文上下文支撑，无依据则不描述。

    ### 2. 对话统计（仅统计魏无羡的直接对话）
    #### 统计规则：
    - ✅ **必须包含**：所有带双引号（“”）的魏无羡口语表达，包括：
      - 与他人互动时的回应；
      - 独自时的自语；
      - 对非人对象（走尸、鬼手、器物等）发出的指令。
    - ❌ **绝对排除**：内心活动（如“魏无羡心想…”）、动作描写、无引号的间接引语。

    #### 输出统计：
    - 魏无羡直接对话总数：【X】条（按原文实际计数）；
    - 连续对话链组数：【Y】组（指“魏无羡说话 → 他人/场景回应 → 魏无羡再说话”的完整链，无则为0）。

    ### 3. 生成微调用JSON（仅含魏无羡对话，场景提示含对话对象泛称）
    #### 输出要求：
    - **仅输出纯标准JSON**，无任何额外文字、注释、Markdown；
    - JSON必须可解析，且样本数量与X/Y严格一致。

    #### JSON结构：
    {{
      "chapter_title": "{title}",
      "samples": [
        // —— 单轮样本（共X条）——
        {{
          "conversations": [
            {{
              "from": "system",
              "value": "你是《魔道祖师》中的魏无羡，请根据当前场景和对话对象，用符合原文语气的方式回应。"
            }},
            {{
              "from": "human",
              "value": "[场景提示：时期 + 地点 + 魏无羡状态 + 对话对象（用泛称）+ 触发事件]"
            }},
            {{
              "from": "assistant",
              "value": "【魏无羡原文对话，一字不改，保留标点与语气词】"
            }}
          ]
        }},
        // —— 多轮样本（共Y组）——
        {{
          "conversations": [
            {{
              "from": "system",
              "value": "你是《魔道祖师》中的魏无羡，请根据当前场景和对话对象，用符合原文语气的方式回应。"
            }},
            {{
              "from": "human",
              "value": "[场景提示1：时期 + 地点 + 状态 + 对话对象泛称 + 触发事件]"
            }},
            {{
              "from": "assistant",
              "value": "【魏无羡第一句原文对话】"
            }},
            {{
              "from": "human",
              "value": "[场景提示2：延续前序场景，提及同一对话对象泛称]"
            }},
            {{
              "from": "assistant",
              "value": "【魏无羡第二句原文对话】"
            }}
          ]
        }}
      ],
      "extraction_note": {{
        "total_dialogues_extracted": X,
        "multi_turn_chains_extracted": Y,
        "covered_scene_types": "包含：有对话对象的互动、无对象的自语/指令（按本章实际覆盖情况描述）"
      }}
    }}

    #### 场景提示编写规范（必须含以下5要素）：
    1. **时期**：根据本章内容判断（如“少年时期”“重生初期”“乱葬岗后期”等）；
    2. **地点**：原文明确地点（如“云深不知处藏书阁”“大梵山山道”）；
    3. **魏无羡状态**：如“被缚时”“饮酒后”“操控阴虎符时”“带伤行走时”；
    4. **对话对象**：**必须包含**，但使用**泛称**，如：
       - “对方”（用于对抗/质疑场景）
       - “同伴”（用于合作/同行场景）
       - “某人”（身份不明时）
       - “年长者”“少年”“修士”等（基于上下文合理推断）
       - 若无对象，则写“独自时”；
    5. **触发事件**：如“对方质问时”“同伴提醒时”“看到异象时”“需操控走尸时”。

    ✅ 正确示例（含对话对象泛称）：
    - “在重生初期的莫家庄院落，魏无羡涂脂抹粉时，面对对方质疑其身份，对方语气不善，请用魏无羡语气说话”
    - “在少年时期的云深不知处后山，魏无羡被罚抄书时，同伴打趣他字迹潦草，请用魏无羡语气说话”
    - “在乱葬岗时期的荒野，魏无羡独自面对失控走尸，需立即发出指令，请用魏无羡语气说话”

    ### 4. 强制校验
    1. **内容忠实**：所有对话来自原文引号内，无增删改；
    2. **阶段真实**：时期与性格描述必须有上下文依据；
    3. **对象合理**：对话对象使用泛称，不出现“蓝忘机”“江澄”等具体名字；
    4. **数量一致**：单轮样本数 = X，多轮样本数 = Y；
    5. **格式纯净**：仅输出可解析JSON，无任何额外字符。

    请严格按此执行，确保数据集**完全基于原文、无预设偏见、保留真实对话语境**，适用于角色语言风格微调。
    """

    try:
        # 动态调整max_tokens（按内容长度估算，1汉字≈2tokens）
        content_length = len(content)
        max_tokens = min(20000, max(2000, content_length * 2))

        response = model.invoke(
            [
                SystemMessage(content="你是专业数据集生成工具..."),
                HumanMessage(content=prompt)
            ],
            config={"max_tokens": max_tokens}  # 参数通过config传递
        )
        # 清理输出（去除可能的格式干扰）
        json_str = response.content.strip()
        json_str = re.sub(r'^```(json)?|```$', '', json_str).strip()  # 去除代码块标记

        # 验证JSON
        chapter_data = json.loads(json_str)

        # 校验结构完整性
        if "chapter_title" not in chapter_data:
            chapter_data["chapter_title"] = title
        if "samples" not in chapter_data:
            chapter_data["samples"] = []

        print(f"章节《{title}》生成 {len(chapter_data['samples'])} 条样本")
        return chapter_data

    except json.JSONDecodeError as e:
        error_msg = f"章节《{title}》JSON解析失败: {str(e)}"
        print(error_msg)
        logging.error(f"{error_msg} 输出内容: {response.content[:500]}")
        return {"chapter_title": title, "samples": []}
    except Exception as e:
        error_msg = f"章节《{title}》处理失败: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return {"chapter_title": title, "samples": []}


def main():
    # 1. 配置与参数
    input_txt = Path("魔道祖师.txt")  # 输入小说文本
    output_json = Path("weiwuxian_dialogues_dataset.json")  # 输出数据集
    test_mode = False  # 测试模式：仅处理前3章
    test_chapter_limit = 3

    # 2. 检查输入文件
    if not input_txt.exists():
        print(f"错误：输入文件 {input_txt} 不存在")
        return

    # 3. 配置模型（从环境变量获取密钥，更安全）
    try:
        model = ChatOpenAI(
            openai_api_base="http://localhost:3000/v1",  # 修正后的API基础路径
            openai_api_key="sk-V5Nou8za8Jaqz7RZ5a501569043846D98e2aA0AaEaCeB6Bb",  # 替换为您的API密钥
            model_name="qwen3-max",  # 确保模型名称正确
            temperature=0.3,  # 降低随机性，确保格式稳定
            #request_timeout=60  # 超时设置
        )
    except Exception as e:
        print(f"模型配置失败: {str(e)}")
        logging.error(f"模型配置失败: {str(e)}")
        return

    # 4. 切分章节
    try:
        chapters = split_chapters(input_txt)
       # for chapter in chapters:
        #    print("*************************************************************")
       #     print(chapter)
       #     print("*************************************************************")

    except Exception as e:
        print(f"章节切分失败: {str(e)}")
        return

    if not chapters:
        print("未识别到有效章节，程序退出")
        return

    # 测试模式限制章节数
    if test_mode:
        chapters = chapters[:test_chapter_limit]
        print(f"测试模式：仅处理前 {len(chapters)} 章")

    # 5. 逐章生成样本（带进度条）
    full_dataset = {"chapters": []}
    for chapter in tqdm(chapters, desc="处理章节"):
        chapter_data = get_model_json_output(chapter, model)
        full_dataset["chapters"].append(chapter_data)

    # 6. 处理输出文件（避免覆盖）
    if output_json.exists():
        backup_path = output_json.with_suffix(f".bak{output_json.suffix}")
        output_json.rename(backup_path)
        print(f"已将原有文件备份至 {backup_path}")

    # 7. 保存结果
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(full_dataset, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存文件失败: {str(e)}")
        logging.error(f"保存文件失败: {str(e)}")
        return

    # 8. 统计结果
    total_samples = sum(len(chap["samples"]) for chap in full_dataset["chapters"])
    print(f"\n处理完成！总样本数：{total_samples}，已保存至 {output_json}")
    print(f"错误日志已记录至 dataset_generation.log")


if __name__ == "__main__":
    main()