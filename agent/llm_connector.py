"""
agent/llm_connector.py
Connexion à Groq API (cloud) ou Ollama (local).
"""

import json
import urllib.request
import urllib.error
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class LLMConnector:

    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self.model    = config.LLM_MODEL
        self.timeout  = config.LLM_TIMEOUT

        if self.provider == "groq":
            self.api_key  = config.GROQ_API_KEY
            self.endpoint = config.GROQ_API_URL
            print(f"[LLM] Groq API — modèle : {self.model}")
        else:
            self.endpoint = f"{config.LLM_BASE_URL}/api/generate"
            print(f"[LLM] Ollama — modèle : {self.model}")

    def call(self, prompt: str, temperature: float = 0.3) -> str:
        if self.provider == "groq":
            return self._call_groq(prompt, temperature)
        else:
            return self._call_ollama(prompt, temperature)

    def _call_groq(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tu es GuideME, assistant pédagogique spécialisé "
                        "UNIQUEMENT dans la procédure de drainage thoracique. "
                        "RÈGLE ABSOLUE : Tu réponds TOUJOURS et UNIQUEMENT en français. "
                        "Jamais en anglais, jamais dans une autre langue. "
                        "Maximum 2 phrases courtes par réponse. "
                        "Tutoie toujours l'étudiant. "
                        "Sois direct, précis et encourageant."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens":  config.AGENT_MAX_TOKENS,
        }
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent":    "Mozilla/5.0",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            print(f"[LLM] Groq erreur HTTP {e.code} : {body}")
            return "Je ne peux pas répondre pour le moment."
        except Exception as e:
            print(f"[LLM] Groq erreur : {e}")
            return "Une erreur s'est produite. Réessaie."

    def _call_ollama(self, prompt: str, temperature: float) -> str:
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {
                "temperature": temperature,
                "num_predict": 150,
                "num_ctx":     512,
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            print(f"[LLM] Ollama erreur : {e}")
            return "Je ne peux pas répondre — vérifie qu'Ollama est lancé."
        except Exception as e:
            print(f"[LLM] Erreur : {e}")
            return "Une erreur s'est produite. Réessaie."

    def is_available(self) -> bool:
        if self.provider == "groq":
            try:
                test = self.call("Réponds juste : ok")
                return bool(test)
            except:
                return False
        else:
            try:
                req = urllib.request.Request(
                    f"{config.LLM_BASE_URL}/api/tags", method="GET"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data   = json.loads(resp.read().decode("utf-8"))
                    models = [m["name"] for m in data.get("models", [])]
                    return any(self.model in m for m in models)
            except:
                return False


if __name__ == "__main__":
    llm = LLMConnector()
    print("Test connexion...")
    r = llm.call("Tu es GuideME. Réponds en français en 1 phrase : qu'est-ce que le drainage thoracique ?")
    print(f"Réponse : {r}")