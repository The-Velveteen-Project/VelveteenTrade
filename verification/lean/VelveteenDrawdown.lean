import VelveteenRisk

/-!
# VelveteenTrade — freno por drawdown con ventana móvil (R31)

Modela el freno de `velveteentrade/risk.py` (`evaluate`, bloque «Freno por
drawdown») con el pico que calcula `Journal.update_window_peak`
(`velveteentrade/memory.py`): el máximo del valor del satélite en las últimas
`W` sesiones, incluida la actual. Antes el pico era el histórico y, tras un
stop a efectivo, el freno no se soltaba nunca (R31).

Teoremas:

* `window_halt_blocks_buy` (seguridad) — mientras la caída respecto al pico de
  la ventana supere el umbral, la puerta (`Velveteen.step`, el modelo de
  `evaluate` de VelveteenRisk.lean) rechaza TODA compra, cualquiera que sea la
  propuesta del LLM y el estado de la cartera sombra.
* `window_halt_releases` (vivacidad) — si en las `W` sesiones de la ventana el
  valor de hoy no está más de `θ` por debajo de ninguno de ellos, el freno está
  suelto. Es la propiedad que R31 violaba.
* `flat_window_releases` — corolario: `W` sesiones con el valor plano (en
  efectivo) sueltan el freno, sea cual sea la historia anterior.
* `r31_all_time_never_releases` — contraejemplo verificado del freno anterior
  (pico histórico): pico 1000, efectivo en 850, umbral 10 %: frenado en TODA
  sesión posterior.
* `r31_window_releases` — la misma serie con ventana `W`: suelto desde la
  sesión `W`.

Solo Lean core (sin Mathlib), sobre ℚ, como VelveteenRisk.lean.

## Qué NO cubre
* Python usa doubles IEEE-754; aquí ℚ. Los NaN/inf no existen en ℚ: en el
  código fallan cerrado (`tests/test_r31_freno_ventana.py`).
* Un valor por sesión. El código guarda el MÁXIMO de los ciclos de cada sesión;
  su pico es ≥ al de este modelo con el último valor, así que la seguridad se
  conserva; la vivacidad se cumple en el caso que importa (efectivo plano).
* Las sesiones son las que tienen registro en el journal (si el bot no corre,
  la ventana abarca más calendario). La reconstrucción desde `equity_history`
  y el reinicio por base nueva (B6c) se prueban en Python, no aquí.
* La correspondencia con Python es por inspección (comentarios `-- memory.py:`
  y `-- risk.py:`), no por extracción mecánica.
-/

namespace Velveteen.Drawdown

/-- `peakBack v t k` = máximo de `v s` para `s` en `[t - k, t]` (resta natural:
la ventana se recorta en la sesión 0 si la historia es más corta).
memory.py: `SELECT MAX(peak) ... ORDER BY session DESC LIMIT window`. -/
def peakBack (v : Nat → Rat) : Nat → Nat → Rat
  | t, 0 => v t
  | 0, _ + 1 => v 0
  | t + 1, k + 1 => max (v (t + 1)) (peakBack v t k)

/-- Todo valor de la ventana está por debajo del pico. -/
theorem le_peakBack (v : Nat → Rat) :
    ∀ (t k s : Nat), t - k ≤ s → s ≤ t → v s ≤ peakBack v t k := by
  intro t
  induction t with
  | zero =>
    intro k s _ h2
    have hs : s = 0 := by omega
    subst hs
    cases k <;> simp [peakBack] <;> grind
  | succ t ih =>
    intro k s h1 h2
    cases k with
    | zero =>
      have hs : s = t + 1 := by omega
      subst hs
      simp [peakBack] <;> grind
    | succ k =>
      simp only [peakBack]
      by_cases hs : s = t + 1
      · subst hs; grind
      · have := ih k s (by omega) (by omega)
        grind

/-- El pico se alcanza en alguna sesión de la ventana. -/
theorem peakBack_attained (v : Nat → Rat) :
    ∀ (t k : Nat), ∃ s, t - k ≤ s ∧ s ≤ t ∧ peakBack v t k = v s := by
  intro t
  induction t with
  | zero =>
    intro k
    refine ⟨0, by omega, by omega, ?_⟩
    cases k <;> simp [peakBack]
  | succ t ih =>
    intro k
    cases k with
    | zero => exact ⟨t + 1, by omega, by omega, by simp [peakBack]⟩
    | succ k =>
      obtain ⟨s, h1, h2, h3⟩ := ih k
      simp only [peakBack]
      by_cases hc : peakBack v t k ≤ v (t + 1)
      · exact ⟨t + 1, by omega, by omega, by grind⟩
      · exact ⟨s, by omega, by omega, by grind⟩

/-- memory.py `update_window_peak`: pico de las últimas `W` sesiones (incluida `t`). -/
def windowPeak (v : Nat → Rat) (W t : Nat) : Rat := peakBack v t (W - 1)

/-- risk.py: `if ctx.peak_equity and current < ctx.peak_equity * (1 - halt)`.
(`ctx.peak_equity` falso en Python = pico 0.) -/
def halted (θ : Rat) (v : Nat → Rat) (W t : Nat) : Prop :=
  windowPeak v W t ≠ 0 ∧ v t < windowPeak v W t * (1 - θ)

instance (θ : Rat) (v : Nat → Rat) (W t : Nat) : Decidable (halted θ v W t) := by
  unfold halted; infer_instance

/-! ## (a) Seguridad -/

/-- Mientras el freno de la ventana está puesto, ninguna compra sale de la
puerta. `vetoed` del modelo de VelveteenRisk.lean es exactamente «freno de la
ventana o filtro CVaR» (risk.py: ambos devuelven el veredicto rechazado).
Cuantifica sobre toda propuesta `d`, todo estado `st`, todo límite `L`. -/
theorem window_halt_blocks_buy {σ κ : Type} [DecidableEq σ] [DecidableEq κ]
    (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ)
    (st : Shadow σ κ) (d : Proposal σ)
    (θ : Rat) (v : Nat → Rat) (W t : Nat) (cvar : Bool)
    (hveto : d.vetoed = (decide (halted θ v W t) || cvar))
    (hh : halted θ v W t) (hbuy : d.act = .buy) :
    step rq E L sec st d = (st, none) := by
  have hv : d.vetoed = true := by simp [hveto, hh]
  simp [step, hbuy, hv]

/-- Forma de la seguridad sin la puerta: el freno puesto equivale a una caída
estrictamente mayor que `θ` respecto al pico de la ventana (pico > 0). -/
theorem halted_iff_drawdown {θ : Rat} {v : Nat → Rat} {W t : Nat}
    (hp : 0 < windowPeak v W t) :
    halted θ v W t ↔ windowPeak v W t * θ < windowPeak v W t - v t := by
  unfold halted
  constructor
  · intro ⟨_, h⟩; grind
  · intro h; exact ⟨by grind, by grind⟩

/-! ## (b) Vivacidad -/

/-- Si en la ventana `[t + 1 - W, t]` el valor de hoy no cae más que `θ`
respecto a ninguna sesión (`(1 - θ) * v s ≤ v t`), el freno está suelto. -/
theorem window_halt_releases {θ : Rat} {v : Nat → Rat} {W t : Nat} (hW : 1 ≤ W)
    (hok : ∀ s, t + 1 - W ≤ s → s ≤ t → (1 - θ) * v s ≤ v t) :
    ¬ halted θ v W t := by
  intro ⟨_, hlt⟩
  obtain ⟨s, h1, h2, h3⟩ := peakBack_attained v t (W - 1)
  have := hok s (by omega) h2
  unfold windowPeak at hlt
  rw [h3, Rat.mul_comm] at hlt
  grind

/-- Corolario: `W` sesiones con el valor plano `c ≥ 0` (en efectivo, por
ejemplo) sueltan el freno, cualquiera que haya sido el pico anterior. -/
theorem flat_window_releases {θ c : Rat} {v : Nat → Rat} {W t : Nat}
    (hW : 1 ≤ W) (hθ : 0 ≤ θ) (hc : 0 ≤ c)
    (hflat : ∀ s, t + 1 - W ≤ s → s ≤ t → v s = c) :
    ¬ halted θ v W t := by
  apply window_halt_releases hW
  intro s h1 h2
  rw [hflat s h1 h2, hflat t (by omega) (Nat.le_refl t)]
  have := Rat.mul_nonneg hθ hc
  grind

/-! ## (c) R31: el freno anterior no se soltaba -/

/-- El freno de antes: pico histórico (memory.py `update_peak_equity`). -/
def allTimeHalted (θ : Rat) (v : Nat → Rat) (t : Nat) : Prop :=
  peakBack v t t ≠ 0 ∧ v t < peakBack v t t * (1 - θ)

/-- Serie de R31: pico 1000 en la sesión 0; stop a efectivo; 850 para siempre. -/
def r31Series (n : Nat) : Rat := if n = 0 then 1000 else 850

theorem r31_peak (t : Nat) : peakBack r31Series t t = 1000 := by
  have hle := le_peakBack r31Series t t 0 (by omega) (by omega)
  obtain ⟨s, _, _, hs⟩ := peakBack_attained r31Series t t
  have h0 : r31Series 0 = 1000 := by simp [r31Series]
  have hs' : r31Series s ≤ 1000 := by unfold r31Series; split <;> grind
  grind

/-- Contraejemplo verificado (R31): con el pico histórico, el freno está
puesto en TODA sesión posterior al stop. Exposición 0 para siempre. -/
theorem r31_all_time_never_releases :
    ∀ t, 1 ≤ t → allTimeHalted (1 / 10) r31Series t := by
  intro t ht
  have hv : r31Series t = 850 := by simp [r31Series]; omega
  unfold allTimeHalted
  rw [r31_peak, hv]
  constructor <;> grind

/-- La misma serie con la ventana: desde la sesión `W` el freno está suelto. -/
theorem r31_window_releases {W : Nat} (hW : 1 ≤ W) :
    ∀ t, W ≤ t → ¬ halted (1 / 10) r31Series W t := by
  intro t ht
  apply flat_window_releases (c := 850) hW (by grind) (by grind)
  intro s h1 _
  simp [r31Series]; omega

end Velveteen.Drawdown
