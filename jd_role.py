"""Generate a scoring role from a natural-language JD.

Keeps the existing role bundle compatible with the project and additionally
writes ``rubric_review.html`` for human review.

Generation flow: JD -> LLM draft -> LLM reviewer -> deterministic validation -> files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from html import escape
from pathlib import Path
from typing import Any

from llm_utils import initialize_llm_provider, extract_json_from_response
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from roles import ROLES_DIR, load_role

logger = logging.getLogger(__name__)

TARGET_WEIGHT = 100
MAX_CATEGORIES = 6
DEFAULT_BONUS_MAX = 10
MAX_BONUS = 20
RUBRIC_VERSION = 2
EDU_KEY = "education_background"
EDU_WEIGHT = 15

DO_NOT_SCORE = [
    "姓名",
    "邮箱、手机号等联系方式",
    "性别",
    "年龄或出生日期",
    "民族、种族",
    "婚姻或家庭状况",
    "照片、外貌",
    "宗教或政治观点",
    "与岗位无关的住址或籍贯",
]

EDUCATION_POLICY = {
    "school": "学校层次可以适度计分：能可靠识别时，可区分985/顶尖双一流、211/双一流、普通本科、专科/其他；无法可靠判断时按中性处理，不得猜测。",
    "major": "专业与岗位直接相关得分更高，相近专业次之，不相关专业较低；只依据简历明确写出的专业。",
    "gpa": "GPA、百分制成绩、专业排名或前百分比可以计分；使用简历原始量表，量表不清时不要强行换算。",
    "missing_gpa": "未写GPA/成绩时不额外扣分，也不得虚构；只是没有GPA这一子项的证据分。",
}

SYSTEM_PROMPT = r"""你是一名资深招聘评分标准设计师。请把自然语言 JD 转成“简历筛选”用的结构化 JSON rubric。

这是辅助招聘人员做初筛的工具，不是自动录用/淘汰系统，最终决定由人工完成。

【公司明确要求】教育背景需要计分：
- 必须包含 education_background 类别，通常占 10-20 分。
- 学校层次、专业相关度、GPA/成绩/排名都可以影响 education_background 得分。
- 学校层次只是总分的一小部分，不应压过与岗位直接相关的技能、项目和工作证据。
- 学校层次无法可靠判断时按中性处理，不得猜测。
- 没写 GPA 不属于扣分，只是没有这一项证据。

【永远不计分】姓名、邮箱/电话、性别、年龄、民族/种族、婚姻家庭、照片外貌、宗教政治观点、与岗位无关的住址籍贯。

【非常重要：只能根据简历可观察证据评分】
- 每个评分标准必须能从简历文本或评分上下文中已经提供的外部数据判断。
- 禁止写“面试中回答不了”“图纸错误很多”“做事马虎”“沟通差”等只有面试/现场/作品检查才能确认的规则，除非简历上下文本身提供了明确证据。
- 缺少证据可以让对应类别少得分，但不能再次触发 deduction，避免重复惩罚。
- bonus 只能奖励正常类别没有重复奖励的额外亮点。
- deduction 只能针对简历中明确存在的负面/矛盾证据；“没有写”本身不能扣分。

只返回 JSON，不要 markdown。结构必须是：
{
  "position_title": "岗位名称",
  "role_key": "lowercase_snake_case",
  "categories": [
    {"key":"technical_skills","label":"技能","max":30,"icon":"🛠️"}
  ],
  "category_criteria": {
    "technical_skills": {
      "summary":"这一类评什么",
      "subcriteria":[
        {"name":"子项","points":15,"observable_evidence":["简历里应看到什么"]}
      ],
      "bands":[
        {"name":"LOW","description":"低分标准","observable_evidence":["证据"]},
        {"name":"MEDIUM","description":"中分标准","observable_evidence":["证据"]},
        {"name":"HIGH","description":"高分标准","observable_evidence":["证据"]}
      ],
      "do_not_infer":["不能自行推断什么"]
    }
  },
  "must_have":["硬要求"],
  "nice_to_have":["加分偏好"],
  "bonus_max":10,
  "bonus_rules":[
    {"rule":"额外亮点","points":2,"evidence_required":"必须看到的证据","non_duplicate_reason":"为什么没有和普通类别重复"}
  ],
  "deduction_rules":[
    {"rule":"明确负面情况","points":2,"evidence_required":"必须看到的明确负面/矛盾证据"}
  ]
}

要求：
- 4-6 个类别，必须包含 education_background。
- 所有类别 max 之和必须恰好为 100。
- 中文 label/说明，英文 snake_case key。
- 每个类别要有 LOW/MEDIUM/HIGH、可观察证据和 do_not_infer。
- 每个类别的 subcriteria points 之和应等于该类别 max。
- bonus_max 通常 5-15，绝不超过 20。
- 不能把同一条证据在普通类别和 bonus 里重复算分。
- 只返回 JSON。"""

REVIEW_PROMPT = r"""你是严格的高级招聘 rubric 审计员。请对照原始 JD 审核草稿，并直接返回“修正后的完整 JSON”，结构和草稿完全一致。

逐项检查：
1. 是否忠于 JD，有没有凭空增加要求。
2. 是否每条规则都能从简历/已提供上下文观察到；删除面试才能判断的内容。
3. 是否存在重复计分：同一证据不能既进入普通类别又当 bonus；缺失信息不能既少得分又 deduction。
4. deduction 必须有明确负面或矛盾证据；单纯没写不能扣分。
5. 必须有 education_background，通常 10-20 分；学校层次、专业相关度、GPA/成绩都要透明地体现在该类里。
6. 学校层次不确定时中性处理；未写 GPA 不扣分。
7. 姓名、联系方式、性别、年龄、民族/种族、婚姻家庭、照片、宗教政治、无关地点永远不计分。
8. 权重之和 100；教育背景不能压过与岗位直接相关的技能/项目/经历。
9. LOW/MEDIUM/HIGH 应清楚、有实际简历证据。
10. 这是人工决策支持，不要生成自动淘汰/录用结论。

只返回修正后的 JSON，不要解释，不要 markdown。"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _safe_slug(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).lower()
    slug = re.sub(r"[-_]{2,}", "-", slug).strip("-_")
    return (slug[:max_len].rstrip("-_")) or "job-description-role"


def _key(text: Any, idx: int, seen: set[str]) -> str:
    k = re.sub(r"[^a-z0-9_]", "_", str(text or "").lower()).strip("_")
    k = re.sub(r"_+", "_", k)
    if not k or k[0].isdigit():
        k = f"category_{idx}"
    base = k
    n = idx
    while k in seen:
        k = f"{base}_{n}"
        n += 1
    seen.add(k)
    return k


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, str):
        out = [x.strip() for x in value.replace(";", "\n").splitlines() if x.strip()]
    elif isinstance(value, list):
        out = [str(x).strip() for x in value if str(x).strip()]
    else:
        out = []
    return out or list(default or [])


def _normalize(items: list[dict[str, Any]], field: str, target: int) -> None:
    total = sum(max(1, int(x[field])) for x in items)
    exact = [max(1, int(x[field])) * target / total for x in items]
    vals = [max(1, int(x)) for x in exact]
    diff = target - sum(vals)
    if diff > 0:
        order = sorted(range(len(vals)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
        for i in range(diff):
            vals[order[i % len(vals)]] += 1
    elif diff < 0:
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        for _ in range(-diff):
            for i in order:
                if vals[i] > 1:
                    vals[i] -= 1
                    break
    for item, value in zip(items, vals):
        item[field] = value


def _education_subcriteria(max_score: int) -> list[dict[str, Any]]:
    items = [
        {"name": "学校/院校层次", "points": 5, "observable_evidence": ["简历明确写出的学校名称；仅在能可靠识别院校层次时适度区分"]},
        {"name": "专业相关度", "points": 6, "observable_evidence": ["简历明确写出的专业及其与岗位的相关度"]},
        {"name": "GPA/成绩/排名", "points": 4, "observable_evidence": ["简历明确给出的GPA、百分制成绩、排名或前百分比"]},
    ]
    _normalize(items, "points", max_score)
    return items


def _fix_subcriteria(value: Any, max_score: int, label: str, education: bool) -> list[dict[str, Any]]:
    out = []
    if isinstance(value, list):
        for i, raw in enumerate(value[:8], 1):
            if not isinstance(raw, dict):
                continue
            out.append({
                "name": str(raw.get("name") or f"子项{i}").strip(),
                "points": _int(raw.get("points"), 1, 1, max_score),
                "observable_evidence": _str_list(raw.get("observable_evidence"), ["简历中的明确证据"]),
            })
    if not out:
        out = [{"name": label, "points": max_score, "observable_evidence": ["简历中与该维度直接相关的明确经历、项目或成果"]}]
    if education:
        names = " ".join(x["name"] for x in out).lower()
        if not (any(x in names for x in ("学校", "院校", "school")) and any(x in names for x in ("专业", "major")) and any(x in names for x in ("gpa", "成绩", "排名"))):
            return _education_subcriteria(max_score)
    _normalize(out, "points", max_score)
    return out


def _bands(value: Any, max_score: int, label: str) -> list[dict[str, Any]]:
    raw_map = {}
    if isinstance(value, list):
        for x in value:
            if isinstance(x, dict):
                n = str(x.get("name") or "").upper()
                if n in {"LOW", "MEDIUM", "HIGH"}:
                    raw_map[n] = x
    low_max = round(max_score * 0.39)
    med_max = min(max_score - 1, max(low_max + 1, round(max_score * 0.74)))
    ranges = {"LOW": (0, low_max), "MEDIUM": (low_max + 1, med_max), "HIGH": (med_max + 1, max_score)}
    defaults = {
        "LOW": f"与{label}相关的直接证据较少，或主要停留在关键词/课程层面。",
        "MEDIUM": f"有与{label}相关的实际证据，但深度、复杂度或独立性一般。",
        "HIGH": f"有充分、具体、直接相关的{label}证据，并体现较强的独立完成能力或实际成果。",
    }
    out = []
    for name in ("LOW", "MEDIUM", "HIGH"):
        raw = raw_map.get(name, {})
        lo, hi = ranges[name]
        out.append({
            "name": name,
            "min": lo,
            "max": hi,
            "description": str(raw.get("description") or defaults[name]).strip(),
            "observable_evidence": _str_list(raw.get("observable_evidence"), ["必须能引用简历或已提供上下文中的具体证据"]),
        })
    return out


def _rules(value: Any, bonus: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out, seen = [], set()
    for raw in value[:12]:
        if isinstance(raw, str):
            rule = raw.strip()
            if not rule:
                continue
            m = re.search(r"(\d+)\s*分?", rule)
            points = int(m.group(1)) if m else 2
            obj = {"rule": rule, "points": points, "evidence_required": "必须能引用简历或已提供上下文中的具体证据"}
        elif isinstance(raw, dict):
            rule = str(raw.get("rule") or raw.get("description") or "").strip()
            if not rule:
                continue
            obj = {
                "rule": rule,
                "points": _int(raw.get("points"), 2, 1, 10),
                "evidence_required": str(raw.get("evidence_required") or "必须能引用简历或已提供上下文中的具体证据").strip(),
            }
        else:
            continue
        norm = re.sub(r"\s+", "", obj["rule"]).lower()
        if norm in seen:
            continue
        seen.add(norm)
        if bonus:
            obj["non_duplicate_reason"] = str(raw.get("non_duplicate_reason") if isinstance(raw, dict) else "" or "只有在普通评分维度尚未奖励同一证据时才适用").strip()
        out.append(obj)
    return out


def _validate(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object")

    raw_categories = data.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("LLM output has no categories")

    categories, seen, key_map = [], set(), {}
    for i, raw in enumerate(raw_categories[:MAX_CATEGORIES], 1):
        if not isinstance(raw, dict):
            continue
        raw_key = str(raw.get("key") or "")
        clean = _key(raw_key, i, seen)
        label = str(raw.get("label") or clean).strip()
        if clean in {"education", "academic_background", "education_score"} or any(x in label for x in ("教育", "学历", "院校")):
            if EDU_KEY not in {c["key"] for c in categories}:
                seen.discard(clean)
                clean = EDU_KEY
                seen.add(clean)
        key_map[raw_key] = clean
        categories.append({"key": clean, "label": label, "max": _int(raw.get("max"), 15, 5, 60), "icon": (str(raw.get("icon") or "•").strip() or "•")[:4]})

    if not any(c["key"] == EDU_KEY for c in categories):
        if len(categories) >= MAX_CATEGORIES:
            idx = min(range(len(categories)), key=lambda i: categories[i]["max"])
            categories[idx] = {"key": EDU_KEY, "label": "教育背景（学校 / 专业 / 成绩）", "max": EDU_WEIGHT, "icon": "🎓"}
        else:
            categories.append({"key": EDU_KEY, "label": "教育背景（学校 / 专业 / 成绩）", "max": EDU_WEIGHT, "icon": "🎓"})

    _normalize(categories, "max", TARGET_WEIGHT)
    edu = next(c for c in categories if c["key"] == EDU_KEY)
    if edu["max"] < 10:
        donor = max((c for c in categories if c["key"] != EDU_KEY), key=lambda c: c["max"])
        move = min(10 - edu["max"], donor["max"] - 1)
        edu["max"] += move; donor["max"] -= move
    elif edu["max"] > 20:
        recv = max((c for c in categories if c["key"] != EDU_KEY), key=lambda c: c["max"])
        recv["max"] += edu["max"] - 20; edu["max"] = 20

    raw_criteria = data.get("category_criteria") if isinstance(data.get("category_criteria"), dict) else {}
    criteria = {}
    for c in categories:
        raw = raw_criteria.get(c["key"])
        if raw is None:
            # try pre-sanitized key aliases
            for k, mapped in key_map.items():
                if mapped == c["key"] and k in raw_criteria:
                    raw = raw_criteria[k]; break
        if not isinstance(raw, dict):
            raw = {}
        education = c["key"] == EDU_KEY
        criteria[c["key"]] = {
            "summary": str(raw.get("summary") or f"根据简历中与“{c['label']}”直接相关、可引用的证据评分。").strip(),
            "subcriteria": _fix_subcriteria(raw.get("subcriteria"), c["max"], c["label"], education),
            "bands": _bands(raw.get("bands"), c["max"], c["label"]),
            "do_not_infer": _str_list(raw.get("do_not_infer"), [
                "不得把只有面试、现场测试或作品检查才能确认的能力当作已知事实",
                "不得因为简历没写某项内容而额外扣分；缺证据只影响该类别得分",
            ]),
        }
        if education:
            for rule in ("无法可靠判断院校层次时不得臆测", "未提供GPA/成绩时不得虚构或额外扣分"):
                if rule not in criteria[c["key"]]["do_not_infer"]:
                    criteria[c["key"]]["do_not_infer"].append(rule)

    return {
        "position_title": str(data.get("position_title") or "岗位").strip(),
        "role_key": _safe_slug(str(data.get("role_key") or "job_description_role")),
        "categories": categories,
        "category_criteria": criteria,
        "must_have": _str_list(data.get("must_have")),
        "nice_to_have": _str_list(data.get("nice_to_have")),
        "bonus_max": _int(data.get("bonus_max"), DEFAULT_BONUS_MAX, 0, MAX_BONUS),
        "bonus_rules": _rules(data.get("bonus_rules"), True),
        "deduction_rules": _rules(data.get("deduction_rules"), False),
        "do_not_score": list(DO_NOT_SCORE),
        "education_policy": dict(EDUCATION_POLICY),
    }


def _criteria_text(data: dict[str, Any]) -> str:
    cats = data["categories"]
    keys = ", ".join(c["key"] for c in cats)
    lines = [
        f"You are evaluating a resume for the position: {data['position_title']}.",
        "This is decision support for a human recruiter, not an automatic hiring decision.", "",
        f"**MANDATORY: fill ALL categories: {keys}.**", "",
        "## NEVER SCORE THESE ATTRIBUTES",
    ]
    lines += [f"- {x}" for x in data["do_not_score"]]
    lines += ["", "## EDUCATION POLICY", "Education is intentionally scoreable for this employer."]
    lines += [f"- School: {data['education_policy']['school']}", f"- Major: {data['education_policy']['major']}", f"- GPA/grades: {data['education_policy']['gpa']}", f"- Missing GPA: {data['education_policy']['missing_gpa']}"]
    lines += ["", "## EVIDENCE RULES", "- Score only resume text or external evidence explicitly supplied in the scoring context.", "- Do not infer interview performance, drawing quality, personality or carelessness without explicit evidence.", "- Missing information may lower the relevant category score but cannot also cause a deduction.", "- Bonus evidence must not duplicate normal-category evidence.", "- Deductions require explicit negative/contradictory evidence; missing information alone is not a deduction."]
    if data["must_have"]:
        lines += ["", "## MUST-HAVE"] + [f"{i}. {x}" for i, x in enumerate(data["must_have"], 1)]
    if data["nice_to_have"]:
        lines += ["", "## NICE-TO-HAVE"] + [f"- {x}" for x in data["nice_to_have"]]
    lines += ["", "## SCORING CRITERIA"]
    for c in cats:
        cr = data["category_criteria"][c["key"]]
        lines += [f"### {c['key']} — {c['label']} (0-{c['max']} points)", cr["summary"], "Subcriteria:"]
        for s in cr["subcriteria"]:
            lines.append(f"- {s['name']}: {s['points']} points")
            lines += [f"  - Observable evidence: {e}" for e in s["observable_evidence"]]
        lines.append("Score bands:")
        for b in cr["bands"]:
            lines.append(f"- {b['name']} ({b['min']}-{b['max']}): {b['description']}")
            lines += [f"  - Evidence: {e}" for e in b["observable_evidence"]]
        lines.append("Do not infer:")
        lines += [f"- {x}" for x in cr["do_not_infer"]]
        lines.append("")
    lines += [f"## BONUS (max {data['bonus_max']})", "Only award bonus when evidence is exceptional and not already counted in a category."]
    lines += [f"- +{x['points']}: {x['rule']} | Evidence required: {x['evidence_required']} | Non-duplicate: {x['non_duplicate_reason']}" for x in data["bonus_rules"]] or ["- No automatic bonus rules."]
    lines += ["", "## DEDUCTIONS", "Do not deduct for missing information."]
    lines += [f"- -{x['points']}: {x['rule']} | Evidence required: {x['evidence_required']}" for x in data["deduction_rules"]] or ["- No automatic deduction rules."]
    lines += ["- Deductions must not make the final score negative.", "", "## OUTPUT", "Respond ONLY with this JSON shape. All evidence fields must cite concrete resume/context evidence or explicitly say relevant evidence is absent.", "{", '  "scores": {']
    for i, c in enumerate(cats):
        comma = "," if i < len(cats) - 1 else ""
        lines.append(f'    "{c["key"]}": {{"score": 0, "max": {c["max"]}, "evidence": "string"}}{comma}')
    lines += ['  },', '  "bonus_points": {"total": 0, "breakdown": "string"},', '  "deductions": {"total": 0, "reasons": "string"},', '  "key_strengths": ["strength1", "strength2"],', '  "areas_for_improvement": ["improvement1"]', "}", "", "Resume to evaluate:", "", "{{ text_content }}"]
    return "\n".join(lines)


def _system_text(data: dict[str, Any], name: str) -> str:
    limits = "\n".join(f"- {c['key']}: 0-{c['max']}" for c in data["categories"])
    return f"""You are an expert recruiter scoring resumes for: {data['position_title']}.
This is decision support for a human recruiter.

Never score name, contact information, gender, age, ethnicity/race, family status, photo, religion/politics or unrelated location.
Education IS intentionally scoreable: school level, major relevance and GPA/academic performance may affect education_background. Unknown school tier is neutral; missing GPA is not a deduction.
Score only observable resume or explicitly supplied external-context evidence. Do not infer interview-only behavior, drawing quality, personality or carelessness. Do not double-count evidence.

Mandatory category limits:
{limits}
Bonus <= {data['bonus_max']}. Deductions must not make final score negative.
Return only valid JSON matching the user prompt. Role key: {name}."""


def _ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(str(x))}</li>" for x in items) + "</ul>" if items else "<p>无</p>"


def _review_html(data: dict[str, Any], name: str, jd_hash: str) -> str:
    blocks = []
    for c in data["categories"]:
        cr = data["category_criteria"][c["key"]]
        subs = "".join(f"<tr><td>{escape(s['name'])}</td><td>{s['points']}</td><td>{_ul(s['observable_evidence'])}</td></tr>" for s in cr["subcriteria"])
        bands = "".join(f"<tr><td>{b['name']}</td><td>{b['min']}-{b['max']}</td><td>{escape(b['description'])}{_ul(b['observable_evidence'])}</td></tr>" for b in cr["bands"])
        blocks.append(f"<section><h2>{escape(c['icon'])} {escape(c['label'])} <span>{c['max']}分</span></h2><p>{escape(cr['summary'])}</p><h3>分值构成</h3><table><tr><th>子项</th><th>分值</th><th>证据</th></tr>{subs}</table><h3>高/中/低档</h3><table><tr><th>档位</th><th>分数</th><th>标准</th></tr>{bands}</table><h3>AI不得推断</h3>{_ul(cr['do_not_infer'])}</section>")
    bonus = "".join(f"<li>+{x['points']}：{escape(x['rule'])}<br><small>证据：{escape(x['evidence_required'])}</small></li>" for x in data["bonus_rules"]) or "<li>无自动加分规则</li>"
    deduction = "".join(f"<li>-{x['points']}：{escape(x['rule'])}<br><small>证据：{escape(x['evidence_required'])}</small></li>" for x in data["deduction_rules"]) or "<li>无自动扣分规则</li>"
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>{escape(data['position_title'])}评分标准</title>
<style>body{{font:15px/1.65 system-ui,'Microsoft YaHei';max-width:1050px;margin:30px auto;padding:0 20px;background:#f6f7fb;color:#222}}section{{background:white;padding:22px;margin:16px 0;border-radius:14px}}h1,h2,h3{{margin-top:0}}h2 span{{float:right;color:#2563eb}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}}.warn{{background:#fff7ed;border-left:4px solid #ea580c;padding:12px}}small{{color:#666}}</style><body>
<section><h1>{escape(data['position_title'])}</h1><p>Role: {escape(name)} · 基础总分100 · Bonus上限{data['bonus_max']}</p><div class='warn'><b>请负责人先确认这份评分标准，再运行简历评分。</b> role.json / jinja 是机器文件，不需要老板阅读。</div></section>
<section><h2>🎓 教育背景政策</h2><p><b>学校、专业、绩点会参与评分；姓名、邮箱、电话等不会参与评分。</b></p>{_ul([EDUCATION_POLICY['school'], EDUCATION_POLICY['major'], EDUCATION_POLICY['gpa'], EDUCATION_POLICY['missing_gpa']])}</section>
<section><h2>必须项</h2>{_ul(data['must_have'])}<h2>偏好项</h2>{_ul(data['nice_to_have'])}</section>
{''.join(blocks)}
<section><h2>⭐ Bonus</h2><ul>{bonus}</ul><h2>⚠️ 扣分</h2><p>“没写”不能单独触发扣分。</p><ul>{deduction}</ul></section>
<section><h2>🚫 永远不参与评分</h2>{_ul(data['do_not_score'])}</section>
<section><h2>老板确认清单</h2><p>□ 权重合理　□ 学校/专业/GPA权重合理　□ must-have真的是硬要求<br>□ Bonus没有重复计分　□ 扣分都有明确负面证据　□ 没有面试才能判断的规则</p><small>JD SHA-256: {escape(jd_hash)}</small></section>
</body></html>"""


def _write(name: str, data: dict[str, Any], jd_hash: str) -> Path:
    role_dir = ROLES_DIR / name
    role_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "position_title": data["position_title"],
        "categories": data["categories"],
        "bonus_max": data["bonus_max"],
        "min_final_score": 0,
        "max_final_score": 100 + data["bonus_max"],
        "generated_from_jd_sha": jd_hash,
        "rubric_version": RUBRIC_VERSION,
        "education_scoring_enabled": True,
    }
    (role_dir / "role.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (role_dir / "criteria.jinja").write_text(_criteria_text(data), encoding="utf-8")
    (role_dir / "system_message.jinja").write_text(_system_text(data, name), encoding="utf-8")
    (role_dir / "rubric_review.html").write_text(_review_html(data, name, jd_hash), encoding="utf-8")
    return role_dir


def _role_exists(name: str) -> bool:
    d = ROLES_DIR / name
    return d.is_dir() and any((d / f).is_file() for f in ("role.json", "criteria.jinja", "system_message.jinja"))


def _find_reusable(jd_hash: str) -> Path | None:
    if not ROLES_DIR.is_dir():
        return None
    for d in ROLES_DIR.iterdir():
        p = d / "role.json"
        if not p.is_file():
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("generated_from_jd_sha") == jd_hash and int(m.get("rubric_version", 0)) == RUBRIC_VERSION:
                return d
        except Exception:
            pass
    return None


def _cat(c: Any, name: str, default: Any = "") -> Any:
    return c.get(name, default) if isinstance(c, dict) else getattr(c, name, default)


def _print_role(name: str, role: Any, role_dir: Path) -> None:
    cats = list(getattr(role, "categories", []) or [])
    weights = ", ".join(f"{_cat(c,'label',_cat(c,'key'))}={_cat(c,'max')}" for c in cats)
    total = sum(int(_cat(c, "max", 0)) for c in cats)
    print(f"✅ role '{name}'（分类总分 {total}）: {weights}")
    print(f"   角色目录：{role_dir}")
    for f in ("role.json", "criteria.jinja", "system_message.jinja", "rubric_review.html"):
        print(f"     - {role_dir / f}")
    print("   👀 请先用浏览器打开 rubric_review.html 让负责人确认，再开始跑简历。")


def _json_call(provider: Any, system: str, user: str, attempts: int = 3) -> dict[str, Any]:
    params = MODEL_PARAMETERS.get(DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last = None
    for attempt in range(attempts):
        try:
            response = provider.chat(
                model=DEFAULT_MODEL,
                messages=messages,
                options={"stream": False, "temperature": min(float(params.get("temperature", 0.1)), 0.2), "top_p": float(params.get("top_p", 0.9))},
                format={},
            )
            text = extract_json_from_response(response["message"]["content"])
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("JSON root is not an object")
            return obj
        except Exception as e:
            last = e
            logger.warning("JD rubric LLM attempt %d/%d failed: %s", attempt + 1, attempts, e)
            messages.append({"role": "user", "content": f"上一次输出无法解析（{e}）。请只返回一个符合要求的完整 JSON 对象。"})
    raise RuntimeError(f"LLM failed after {attempts} attempts: {last}")


def generate_role_from_jd(jd_path: str | Path, role_key: str | None = None, force: bool = False):
    """Generate role files only. This function never scans or scores resumes."""
    jd_path = Path(jd_path)
    if not jd_path.is_file():
        raise FileNotFoundError(f"JD file not found: {jd_path}")
    jd_text = jd_path.read_text(encoding="utf-8").strip()
    if not jd_text:
        raise ValueError(f"JD file is empty: {jd_path}")

    jd_hash = _sha256(jd_text)
    existing = _find_reusable(jd_hash)
    if existing and not force:
        role = load_role(existing.name)
        print(f"♻️  Reusing existing role '{existing.name}' (same JD + rubric v{RUBRIC_VERSION}).")
        _print_role(existing.name, role, existing)
        return role

    requested = _safe_slug(role_key) if role_key else None
    if requested and _role_exists(requested) and not force:
        raise ValueError(f"Role '{requested}' already exists at {ROLES_DIR / requested}; pass --force to regenerate.")

    provider = initialize_llm_provider(DEFAULT_MODEL)
    print(f"🤖 [1/2] 正在用 {DEFAULT_MODEL} 根据 JD 生成评分标准草稿...")
    draft = _validate(_json_call(provider, SYSTEM_PROMPT, f"请根据下面 JD 生成 rubric：\n\n{jd_text}"))

    print("🔎 [2/2] 正在审计可观察证据、重复计分、学校/专业/GPA权重和扣分规则...")
    try:
        reviewed = _json_call(provider, REVIEW_PROMPT, f"原始 JD：\n{jd_text}\n\n草稿：\n{json.dumps(draft, ensure_ascii=False, indent=2)}")
        data = _validate(reviewed)
    except Exception as e:
        logger.warning("Rubric reviewer failed; using validated draft: %s", e)
        print(f"⚠️ reviewer失败，改用已通过程序校验的第一版：{e}")
        data = draft

    name = _safe_slug(requested or data.get("role_key") or "job-description-role")
    if _role_exists(name) and not force:
        raise ValueError(f"Role '{name}' already exists at {ROLES_DIR / name}; pass --force to regenerate.")

    role_dir = _write(name, data, jd_hash)
    role = load_role(name)
    _print_role(name, role, role_dir)
    return role


# Backward-compatible helper for any code that still imports the old private name.
def _call_llm(jd_text: str) -> str:
    provider = initialize_llm_provider(DEFAULT_MODEL)
    draft = _validate(_json_call(provider, SYSTEM_PROMPT, f"请根据下面 JD 生成 rubric：\n\n{jd_text}"))
    try:
        data = _validate(_json_call(provider, REVIEW_PROMPT, f"原始 JD：\n{jd_text}\n\n草稿：\n{json.dumps(draft, ensure_ascii=False, indent=2)}"))
    except Exception:
        data = draft
    return json.dumps(data, ensure_ascii=False)
