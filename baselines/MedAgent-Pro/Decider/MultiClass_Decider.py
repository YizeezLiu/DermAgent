"""
Multi-class final decision module for dermatology diagnosis (Task 1).

Unlike Pro_Decider (binary: Positive/Negative), this module selects
exactly one diagnosis from a candidate list based on collected indicators.
"""

import json
import os
import base64
from openai import OpenAI


class MultiClass_Decider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _safe_json_parse(self, text: str) -> dict:
        t = (text or "").strip()
        if t.startswith("```"):
            parts = sorted((p.strip() for p in t.split("```")), key=len, reverse=True)
            for p in parts:
                if p.startswith("{"):
                    try:
                        return json.loads(p)
                    except Exception:
                        pass
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            s, e = t.find("{"), t.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(t[s : e + 1])
            raise

    def decide(
        self,
        output_file: str,
        prompt: str,
        indicators: list,
        candidate_classes: list,
        image_paths=None,
        field: str = "overall",
    ) -> dict:
        """
        Ask GPT-4o to pick the most likely diagnosis from *candidate_classes*
        given the collected indicator evidence.

        Args:
            output_file: path to write JSON result
            prompt: task prompt (already includes input/goal)
            indicators: [{'indicator_name': str, 'evidence': str}, ...]
            candidate_classes: list of valid class names
            image_paths: optional original image(s) for visual grounding
            field: key in the output JSON

        Returns:
            dict written to output_file
        """
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]

        # Format indicator evidence
        lines = []
        for it in indicators:
            name = str(it.get("indicator_name", "")).strip()
            ev = it.get("evidence", it.get("if_abnormal", ""))
            if isinstance(ev, (dict, list)):
                ev = json.dumps(ev, ensure_ascii=False)
            lines.append(f"- {name}: {ev}")
        ind_text = "Collected diagnostic evidence:\n" + "\n".join(lines)

        classes_str = ", ".join(candidate_classes)
        system_msg = (
            "You are a board-certified dermatologist making a final diagnosis.\n"
            "Given the diagnostic evidence from multiple analysis tools, select "
            "exactly ONE diagnosis from the candidate list.\n"
            "Return ONLY a JSON object with keys:\n"
            '  "diagnosis" (string — must be one of the candidates),\n'
            '  "confidence" (float 0-1),\n'
            '  "reasoning" (brief string).'
        )

        image_messages = []
        if image_paths:
            for path in image_paths:
                b64 = self.encode_image(path)
                image_messages.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

        user_content = image_messages + [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n\n"
                    f"{ind_text}\n\n"
                    f"Candidate diagnoses: [{classes_str}]\n\n"
                    "Return ONLY the JSON object. "
                    '"diagnosis" MUST be exactly one of the candidates above.'
                ),
            }
        ]

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        completion = self.client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=messages,
        )
        raw = completion.choices[0].message.content

        try:
            obj = self._safe_json_parse(raw)
        except Exception:
            obj = {"diagnosis": raw.strip(), "confidence": 0.0, "reasoning": "parse_error"}

        diag = str(obj.get("diagnosis", "")).strip().lower()
        candidates_lower = {c.lower(): c for c in candidate_classes}
        matched = candidates_lower.get(diag, None)
        if matched is None:
            for cl, co in candidates_lower.items():
                if cl in diag or diag in cl:
                    matched = co
                    break
        if matched is None:
            matched = obj.get("diagnosis", "unknown")

        result_obj = {
            "diagnosis": matched,
            "confidence": float(obj.get("confidence", 0.0)),
            "reasoning": str(obj.get("reasoning", "")),
            "raw": raw,
        }

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = {}

        existing[field] = result_obj
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

        return existing
