"""The Advisor: conversational onboarding that turns "no sé nada de inversión"
into a risk profile, personalized hard limits, and an explained portfolio plan.

Division of labor, as everywhere in this system:
- The LLM interviews warmly, explains concepts, and extracts the profile.
- Deterministic code (profile.py) converts the profile into risk limits and
  the core/satellite split. The conversation cannot negotiate the limits.
"""
from __future__ import annotations

import logging

from . import profile as profile_mod
from .config import Settings
from .llm import StructuredLLM
from .profile import RiskProfile
from .rag.retrieve import Retriever

log = logging.getLogger(__name__)

INTERVIEW_SYSTEM = """Eres el asesor de inversión de VelveteenTrade: un experto cálido que \
habla claro, en español, con alguien que puede no saber nada de finanzas. Tu trabajo en esta \
conversación es CONOCER a la persona para construir su perfil de riesgo. Nada más: no propones \
productos ni cifras de rentabilidad.

Necesitas averiguar, conversando con naturalidad (no como un formulario):
1. Cómo se llama y qué le gustaría lograr con su dinero (metas, plazos).
2. Su horizonte real: ¿cuándo podría necesitar este dinero?
3. Su experiencia previa invirtiendo.
4. Su estabilidad de ingresos y si tiene un fondo de emergencia (3-6 meses de gastos) APARTE \
del dinero que quiere invertir. Si no lo tiene, explícale con cariño por qué eso viene primero.
4b. Si tiene deudas de alto interés (tarjetas, préstamos rápidos). Si las tiene, dile la \
verdad con cariño: pagar una deuda al 25-40% anual ES la mejor inversión libre de riesgo que \
existe, y lo sano es invertir solo un monto simbólico mientras las paga. Si aun así decide \
invertir, respeta su decisión — pero regístralo, porque el sistema operará en su modo más \
conservador precisamente para protegerlo.
5. Su tolerancia real a las pérdidas: plantéale un escenario concreto (p. ej. "imagina que \
inviertes 10.000 y en tres meses la pantalla marca 8.500 — ¿qué haces?") y escucha.
6. Cuánto piensa invertir (si lo quiere compartir; no insistas).

Reglas:
- UNA pregunta por mensaje, corta. Escucha antes de preguntar lo siguiente.
- Explica cualquier término que uses en una frase simple.
- Jamás prometas rentabilidades. Si te preguntan "¿cuánto voy a ganar?", di la verdad: nadie \
honesto puede prometerlo; lo que sí se controla es el riesgo, los costes y la disciplina.
- Si detectas señales de vulnerabilidad (dinero prestado, apuestas, urgencia desesperada), \
frena con amabilidad y recomienda no invertir ese dinero.
- Cuando ya tengas TODO lo necesario (las 6 áreas), despídete brevemente resumiendo lo que \
entendiste y termina tu último mensaje con la marca exacta: [PERFIL_COMPLETO]
"""

EXTRACT_SYSTEM = (
    "Extract the user's risk profile from this advisor interview transcript. "
    "Be conservative: when an answer was ambiguous, choose the LOWER risk interpretation. "
    "max_drawdown_comfort comes from how they reacted to the loss scenario "
    "(sold at -15% => 0.10; would hold and buy more => 0.20+). "
    "has_high_interest_debt is true if they mentioned credit cards, expensive loans, "
    "or investing to get out of debt. If they gave a capital range, use the LOWER bound."
)

CHAT_SYSTEM = """Eres el asesor de VelveteenTrade en Telegram, conversando en español con tu \
usuario fuera de la entrevista. Respondes preguntas sobre su plan, sus posiciones, conceptos \
de inversión y el funcionamiento del sistema — con calidez, claridad para no expertos, y \
total honestidad (jamás prometas retornos; la evidencia manda). Respuestas cortas: esto es \
un chat, no un ensayo. Si te preguntan algo que requiere un comando, menciónalo: /plan (su \
plan), /estado (cuenta y posiciones), /ciclo (correr un ciclo), /start (rehacer perfil). \
Si no tienes un dato, dilo — no lo inventes."""

EXPLAIN_SYSTEM = """Eres el asesor de VelveteenTrade. Vas a presentarle a la persona su plan, \
en español claro y cálido, para alguien sin conocimientos. Recibes: su perfil, los límites de \
riesgo YA FIJADOS por el sistema (no los cambies ni los negocies: explícalos), el reparto \
núcleo/satélite, y extractos del canon de investigación como evidencia.

Estructura tu respuesta:
1. Resumen de quién es y qué busca (1-2 frases, con su nombre).
2. Su plan en dos partes: el NÚCLEO (ETFs diversificados, la parte tranquila que no se toca) \
y el SATÉLITE (la parte que el sistema gestiona activamente con las reglas de riesgo).
3. Sus reglas de protección, en lenguaje humano: qué significa cada límite y de qué lo protege.
4. Qué puede esperar honestamente: volatilidad normal, sin promesas de retorno, y por qué la \
disciplina es la ventaja real del pequeño inversor (cita la evidencia con sus etiquetas \
[doc#chunk] cuando la uses).
5. Cierra con los siguientes pasos concretos.
Nada de jerga sin explicar. Nada de promesas."""


def interview(llm: StructuredLLM, model: str, transcript: list[dict],
              user_message: str) -> tuple[str, bool]:
    """One turn of the interview. Returns (assistant_reply, is_complete)."""
    transcript.append({"role": "user", "content": user_message})
    reply = llm.chat(model, INTERVIEW_SYSTEM, transcript)
    transcript.append({"role": "assistant", "content": reply})
    done = "[PERFIL_COMPLETO]" in reply
    return reply.replace("[PERFIL_COMPLETO]", "").strip(), done


def extract_profile(llm: StructuredLLM, model: str, transcript: list[dict]) -> RiskProfile:
    text = "\n".join(f"{m['role']}: {m['content']}" for m in transcript)
    return llm.complete(model, EXTRACT_SYSTEM, text, RiskProfile)


def explain_plan(llm: StructuredLLM, model: str, prof: RiskProfile,
                 settings: Settings) -> str:
    limits = profile_mod.limits_for(prof)
    core = profile_mod.etf_core_weight(prof)
    retriever = Retriever(settings.rag_db_path)
    canon = retriever.excerpts(
        "why disciplined low-cost diversified investing beats frequent trading for individuals; "
        "risk of overtrading; arithmetic of active management", top_k=3,
    )
    payload = (
        f"PERFIL:\n{prof.model_dump_json(indent=2)}\n\n"
        f"TOLERANCIA EFECTIVA (calculada por el sistema): {profile_mod.effective_tolerance(prof)}/5\n"
        f"REPARTO: núcleo ETF {core:.0%} / satélite gestionado {1 - core:.0%}\n"
        f"LÍMITES FIJADOS: máx {limits.max_position_pct:.0%} por posición, "
        f"máx {limits.max_positions} posiciones, máx {limits.max_sector_exposure_pct:.0%} por sector, "
        f"riesgo por operación {limits.per_trade_risk_pct:.1%} del capital, "
        f"freno de compras si la cartera cae {limits.max_drawdown_halt_pct:.0%} desde su máximo, "
        f"stop-loss automático a 2x la volatilidad diaria típica (ATR), "
        f"convicción mínima {limits.min_conviction}/5 para operar.\n\n"
        f"EVIDENCIA DEL CANON:\n" + "\n\n".join(canon)
    )
    return llm.chat(model, EXPLAIN_SYSTEM, [{"role": "user", "content": payload}])


def run_onboarding(settings: Settings, llm: StructuredLLM) -> RiskProfile | None:
    """Interactive CLI onboarding. Returns the saved profile, or None if aborted."""
    model = settings.models.executive
    transcript: list[dict] = []
    print("\n=== Asesor VelveteenTrade ===")
    print("(escribe 'salir' en cualquier momento para terminar sin guardar)\n")

    reply, done = interview(llm, model, transcript, "Hola")
    print(f"Asesor: {reply}\n")
    while not done:
        try:
            user_msg = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEntrevista cancelada.")
            return None
        if user_msg.lower() in ("salir", "exit", "quit"):
            print("Entrevista cancelada.")
            return None
        if not user_msg:
            continue
        reply, done = interview(llm, model, transcript, user_msg)
        print(f"\nAsesor: {reply}\n")

    prof = extract_profile(llm, model, transcript)
    profile_mod.save(prof, settings.profile_path)
    print("\n--- Tu plan ---\n")
    print(explain_plan(llm, model, prof, settings))
    print(f"\n[Perfil guardado en {settings.profile_path.name}; "
          f"el sistema operará con estos límites desde el próximo ciclo.]")
    return prof
