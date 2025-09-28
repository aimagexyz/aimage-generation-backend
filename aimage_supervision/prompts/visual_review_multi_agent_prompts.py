# -*- coding: utf-8 -*-
from typing import Optional


def build_guideline_prompt(
    rpd_type: str,
    base_prompt: str,
    description_for_ai: str,
    image_count: int,
) -> str:
    """ 使用动态上下文和标签规则构建统一的指南提示 （guideline prompt）"""

    if image_count <= 0:
        image_context = """
                ## 参照画像
                参照画像は提供されていません。説明文のみを基にガイドラインを整理してください。
                """
    elif image_count == 1:
        image_context = """
                ## 参照画像
                提供された1枚の画像を詳細に分析し、説明文と照らし合わせてガイドラインを構築してください。
                """
    else:
        image_context = f"""
                ## 参照画像
                提供された{image_count}枚の画像を相互に比較し、共通点と許容される差異を整理してください。
                """

    if rpd_type == 'classification tasks':
        label_guidance = """
                ## ラベル抽出要件
                - 「## 検証タスク」で提示された説明文を精読し、分類候補として列挙されている正式名称をすべて抽出してください。
                - 箇条書き、コロン区切り、または「」で囲まれた語句など、明示的に提示されたバージョン名／カテゴリ名を順番通りに拾い上げます。
                - 説明の末尾にある「必須出力形式」などで再掲される候補も含め、記載されている全ての正式名称を漏れなく抽出します。
                - 文字列の前後から記号（「」やコロン、括弧など）を取り除き、原文の表記そのものを維持してください（例：「「2018最初版」: …」→「2018最初版」）。
                - 新しいラベルを想像で追加したり、意味の異なる言い換えに置き換えてはいけません。
                - `labels` 配列は最低でも1件を返す必要があります。見落としがないか必ず再確認してください。
                """
        labels_snippet = '"labels": ["抽出したラベル1", "抽出したラベル2", "抽出したラベル3"]'
    else:
        label_guidance = """
                ## ラベル要件
                - このタスクは合規／適合判定です。
                - `labels` 配列は必ず ["safe", "alert", "risk"] の順序で固定してください。
                """
        labels_snippet = '"labels": ["safe", "alert", "risk"]'

    return f"""
            あなたは日本のアニメ制作における視覚的品質管理の専門家です。

            ## 基本指針
            {base_prompt}

            ## 検証タスク
            {description_for_ai}

            {label_guidance}  
            
            {image_context}

            ## 出力形式（必ず次のJSONオブジェクト形式で出力してください）
            以下のキー構成を持つJSONオブジェクトを返してください（配列のみの出力は不可）：

            {{
              "visual_guidelines": [
                {{
                  "category": "カテゴリ名",
                  "criteria": "具体的な判定基準",
                  "reference_features": "参照特徴（必要に応じて）",
                  "expected_result": "期待される視覚的結果"
                }}
              ],
              "detailed_analysis": {{
                "colors": ["主要色彩の記述"],
                "shapes": ["重要な形状要素"],
                "proportions": ["比例関係の記述"],
                "textures": ["質感・スタイルの記述"]
              }},
              "comparison_analysis": {{
                "consistencies": ["画像間で一貫している要素"],
                "variations": ["許容される差異"],
                "critical_features": ["最重要の判定要素"]
              }},
              "key_checkpoints": [
                "重要チェックポイント1",
                "重要チェックポイント2"
              ],
              {labels_snippet}
            }}

            ## 厳格な出力規則
            - 出力は上記のJSONオブジェクトのみ。
            - 追加の説明・余計なテキストを含めない。
            - Markdownのコードブロック（```）で囲まない。
            """


def build_constraint_prompt(
    rpd_title: str,
    guidelines_json_str: str,
) -> str:
    return f"""
              以下のガイドラインをAI品質検査で検証可能な制約条件（constraints）へ変換してください。

              ## レビュー項目タイトル
              {rpd_title}

              ## 入力ガイドライン（JSON）
              {guidelines_json_str}

              ## 出力形式（JSONのキーは英語のまま）
              次の形式でJSONを出力してください：

              {{
                "verification_constraints": [
                  {{
                    "constraint_id": "unique_id",
                    "category": "verification_category",
                    "check_method": "検証手順（日本語）",
                    "success_criteria": "成功条件（日本語）",
                    "failure_indicators": "失敗の兆候（日本語）",
                    "severity_mapping": {{
                      "critical": "risk",
                      "moderate": "alert",
                      "minor": "safe"
                    }}
                  }}
                ],
                "priority_order": ["constraint_id_list"]
              }}
              """


def build_enum_prompt(
    rpd_title: str,
    task_description: str,
    constraints_json_str: str,
    bbox_text: Optional[str],
) -> str:
    bbox_section = f"## 重要エリア\n{bbox_text}" if bbox_text else ""
    return f"""
            あなたはアニメーション品質検査の第一次スクリーニング担当者です。

            ## レビュー項目タイトル
            {rpd_title}

            ## タスクの説明（全文）
            {task_description}

            ## 検証制約（JSON）
            {constraints_json_str}

            {bbox_section}

            ## 厳格な出力規則（必須）
            - 出力は次のJSONオブジェクトのみ。
            - 追加の説明や余計なテキストを含めない。
            - Markdownのコードブロック（```）で囲まない。
            - キー名は正確に "label" のみ。
            - 判定に迷う場合は必ず {{"label":"likely_safe"}} を返す。

            ## 出力形式（厳守）
            {{
              "label": "potential_issues" または "likely_safe"
            }}
            """


def build_evidence_prompt(
    rpd_title: str,
    constraints_json_str: str,
    bbox_text: Optional[str],
    base_prompt: Optional[str] = None,
) -> str:
    bbox_section = f"## 重要エリア\n{bbox_text}" if bbox_text else ""
    base_section = f"## 基本指針\n{base_prompt}" if base_prompt else ""
    return f"""
            あなたは経験豊富なアニメーション品質管理専門家です。

            ## タスク
            "{rpd_title}" に関して、以下の制約条件に基づいて画像を詳細に検証し、
            品質判定を行ってください。

            {base_section}

            ## 検証制約（JSON）
            {constraints_json_str}

            {bbox_section}

            ## 厳格な出力規則
            - 出力は次のJSONオブジェクトのみ。
            - 追加の説明・余計なテキストを含めない。
            - Markdownのコードブロック（```）で囲まない。

            ## 出力形式
            以下のJSONフォーマットで結果を返してください：

            {{
              "severity": "risk/alert/safe のいずれか",
              "confidence": 0.0-1.0の信頼度,
              "description": "日本語での詳細説明",
              "suggestion": "日本語での改善提案（問題がある場合のみ）",
              "evidence": [
                "具体的な証拠や根拠"
              ]
            }}
            """


def build_classification_prompt(
    rpd_title: str,
    constraints_json_str: str,
    bbox_text: Optional[str],
    base_prompt: Optional[str] = None,
    allowed_labels: Optional[list[str]] = None,
) -> str:
    bbox_section = f"## 重要エリア\n{bbox_text}" if bbox_text else ""
    base_section = f"## 基本指針\n{base_prompt}" if base_prompt else ""
    labels_section = ""
    if allowed_labels:
        labels_joined = "\n".join([f"- {lbl}" for lbl in allowed_labels])
        labels_section = f"\n## 候補ラベル（必ずこの中から1つを選択）\n{labels_joined}\n"
    return f"""
            あなたは経験豊富なアニメーション品質管理・ラベリングの専門家です。

            ## タスク
            "{rpd_title}" に関して、以下の分類仕様（constraints）に従い、
            入力画像を最も適切なクラスに分類してください。

            {base_section}

            ## 分類仕様（JSON）
            {constraints_json_str}

            {bbox_section}
            {labels_section}

            ## 厳格な出力規則
            - 出力は次のJSONオブジェクトのみ。
            - 追加の説明・余計なテキストを含めない。
            - Markdownのコードブロック（```）で囲まない。

            ## 出力形式
            以下のJSONフォーマットで結果を返してください：

            {{
              "label": "最終ラベル（文字列）",
              "confidence": 0.0-1.0の信頼度,
              "rationale": "判断根拠（簡潔に）"
            }}
            """


def build_label_set_prompt(
    rpd_title: str,
    description_for_ai: str,
) -> str:
    return f"""
            あなたは経験豊富なアニメーション品質管理・ラベリングの専門家です。

            ## タスク
            次の説明文から、分類に用いる候補ラベル（正式名称）を抽出してください。

            ## description_for_ai（全文）
            {description_for_ai}

            ## 出力要件
            - ラベルは順序を保った配列で出力し、ダブりや似たラベルは統一してください。
            - ラベルは人が読む表示名（例：「2018最初版」「2022最新版」「柱稽古編#6から」）。
            - 余計な説明は不要、JSON以外のテキストを出力しないこと。

            ## 出力形式（厳守）
            {{
              "labels": ["ラベル1", "ラベル2", "ラベル3"],
              "notes": "（任意）補足があれば簡潔に"
            }}
            """
