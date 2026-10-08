/-!
# VelveteenTrade — formal model of the deterministic risk gate

This file models `velveteentrade/risk.py` (function `evaluate` and the shadow
portfolio kept in `CycleContext`) over exact rationals, and proves:

* `buy_safe`, `step_preserves` — one decision never pushes a cap over its limit
  (invariants G1, G3 in risk.py).
* `buy_risk_le_budget`, `stop_loss_le_budget` — an approved BUY risks at most
  its budget between the reference price and its stop (G2), when the stop is
  frozen at entry and the exit happens at or above it (no gap).
* `cycle_safe` — over ANY list of proposals (any output of the LLM agents, any
  market data) a portfolio that starts within its caps ends within its caps (C).
* `old_gate_unsafe` — the gate as it was before Oct 2026 (caps checked against
  the broker snapshot, not the shadow portfolio) violates `cycle_safe`: a
  machine-checked counterexample.

Only Lean core is used (no Mathlib): `Rat`, its order lemmas, `grind`, `omega`.

## What is NOT covered (read before citing these theorems)
* Python computes in IEEE-754 doubles; the proofs are over ℚ. The fuzz test
  `tests/test_risk_cycle.py` checks the same invariants on the float code.
* The model is hand-written from the Python source; the correspondence is by
  inspection (comments `-- risk.py:` below), not mechanically extracted.
* Fills are not modeled: orders are assumed to fill at the reference close.
  Gaps through the stop and slippage are outside `stop_loss_le_budget`.
* The drawdown halt and the CVaR filter only ever reject; they are abstracted
  as the input flag `vetoed`.
-/

namespace Velveteen

/-! ## Rounding -/

/-- Specification of quantity rounding: never negative and, when positive,
never above its argument (rounding down / toward zero). -/
structure RoundDown (rq : Rat → Rat) : Prop where
  nonneg : ∀ x, 0 ≤ rq x
  le_of_pos : ∀ x, 0 < rq x → rq x ≤ x

/-- risk.py: `_round_qty(x) = max(0, floor(x * 1000) / 1000)`. -/
def roundQty (x : Rat) : Rat := max 0 (((x * 1000).floor : Rat) / 1000)

theorem roundQty_spec : RoundDown roundQty := by
  constructor
  · intro x
    unfold roundQty
    grind
  · intro x h
    unfold roundQty at *
    have hf := Rat.floor_le (x * 1000)
    grind

/-- risk.py: `_stop_price(close, d) = ceil((close - d) * 100) / 100`. -/
def stopPrice (p sd : Rat) : Rat := (((p - sd) * 100).ceil : Rat) / 100

theorem stopPrice_ge (p sd : Rat) : p - sd ≤ stopPrice p sd := by
  unfold stopPrice
  have h : (p - sd) * 100 ≤ (((p - sd) * 100).ceil : Rat) := Rat.le_ceil
  grind

/-! ## Two arithmetic facts used everywhere -/

theorem mul_le_of_le_div {a b p : Rat} (hp : 0 < p) (h : a ≤ b / p) : a * p ≤ b := by
  have h1 := Rat.mul_le_mul_of_nonneg_right h (Rat.le_of_lt hp)
  rwa [Rat.div_mul_cancel (Rat.ne_of_gt hp)] at h1

theorem div_lt_of_lt_mul {a b p : Rat} (hp : 0 < p) (h : b < a * p) : b / p < a := by
  apply Rat.lt_of_mul_lt_mul_right (c := p) _ (Rat.le_of_lt hp)
  rwa [Rat.div_mul_cancel (Rat.ne_of_gt hp)]

/-! ## The gate -/

structure Limits where
  posCap       : Rat   -- max_position_pct
  maxPositions : Nat   -- max_positions
  grossCap     : Rat   -- max_gross_exposure
  riskPct      : Rat   -- per_trade_risk_pct
  stopMult     : Rat   -- stop_atr_mult
  turnCap      : Rat   -- max_daily_turnover_pct
  sectorCap    : Rat   -- max_sector_exposure_pct
  minConv      : Nat   -- min_conviction
  minNotional  : Rat   -- MIN_ORDER_NOTIONAL

inductive Action | buy | sell | hold
  deriving DecidableEq

/-- One proposal reaching the gate. `sym`, `act`, `conv` come from the LLM;
`close`, `atr`, `heldQty`, `vetoed` come from market data and the broker.
Every theorem below quantifies over ALL values of every field. -/
structure Proposal (σ : Type) where
  sym     : σ
  act     : Action
  conv    : Nat
  close   : Rat
  atr     : Rat
  heldQty : Rat    -- broker quantity of `sym` (used only to size a SELL)
  vetoed  : Bool   -- drawdown halt or CVaR filter fired

/-- risk.py `CycleContext`: the shadow portfolio and the cycle budgets. -/
structure Shadow (σ κ : Type) where
  held    : List σ     -- keys of held_notional
  expo    : σ → Rat    -- held_notional
  gross   : Rat        -- gross
  secExpo : κ → Rat    -- sector_notional
  turn    : Rat        -- turnover_used

variable {σ κ : Type} [DecidableEq σ] [DecidableEq κ]

def upd {α β : Type} [DecidableEq α] (f : α → β) (a : α) (b : β) : α → β :=
  fun x => if x = a then b else f x

/-- risk.py: `if base + qty*close > cap: qty = _round_qty((cap - base) / close)`. -/
def trim (rq : Rat → Rat) (cap base p q : Rat) : Rat :=
  if cap < base + q * p then rq ((cap - base) / p) else q

/-- The BUY size after all three caps (position, gross, sector). -/
def buySize (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ)
    (st : Shadow σ κ) (d : Proposal σ) : Rat :=
  -- q0: risk budget / stop distance; then position, gross and sector caps.
  trim rq (L.sectorCap * E) (st.secExpo (sec d.sym)) d.close
    (trim rq (L.grossCap * E) st.gross d.close
      (trim rq (L.posCap * E) (st.expo d.sym) d.close
        (rq (E * L.riskPct * ((d.conv : Rat) / 5) / (L.stopMult * d.atr)))))

/-- Portfolio after an approved SELL of `d.sym` (risk.py `_apply_sell`). -/
def sellState (sec : σ → κ) (st : Shadow σ κ) (d : Proposal σ) : Shadow σ κ :=
  { held := st.held.erase d.sym, expo := upd st.expo d.sym 0,
    gross := max 0 (st.gross - st.expo d.sym),
    secExpo := upd st.secExpo (sec d.sym) (max 0 (st.secExpo (sec d.sym) - st.expo d.sym)),
    turn := st.turn + d.heldQty * d.close }

/-- Portfolio after an approved BUY of notional `n` (risk.py `_apply_buy`). -/
def buyState (sec : σ → κ) (st : Shadow σ κ) (d : Proposal σ) (n : Rat) : Shadow σ κ :=
  { held := if d.sym ∈ st.held then st.held else d.sym :: st.held,
    expo := upd st.expo d.sym (st.expo d.sym + n),
    gross := st.gross + n,
    secExpo := upd st.secExpo (sec d.sym) (st.secExpo (sec d.sym) + n),
    turn := st.turn + n }

/-- One call of risk.py `evaluate`. Verdict: `some q` = approved with
quantity `q`, `none` = rejected. -/
def step (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ)
    (st : Shadow σ κ) (d : Proposal σ) : Shadow σ κ × Option Rat :=
  match d.act with
  | .hold => (st, none)
  | .sell =>
    if d.sym ∉ st.held ∨ d.heldQty ≤ 0 then (st, none)
    else if L.turnCap * E < st.turn + d.heldQty * d.close then (st, none)
    else (sellState sec st d, some d.heldQty)
  | .buy =>
    if d.conv < L.minConv ∨ d.vetoed then (st, none)
    else if d.sym ∉ st.held ∧ L.maxPositions ≤ st.held.length then (st, none)
    else if d.atr ≤ 0 ∨ d.close ≤ 0 then (st, none)
    else if buySize rq E L sec st d ≤ 0 ∨
        buySize rq E L sec st d * d.close < L.minNotional then (st, none)
    else if L.turnCap * E < st.turn + buySize rq E L sec st d * d.close then (st, none)
    else (buyState sec st d (buySize rq E L sec st d * d.close), some (buySize rq E L sec st d))

def run (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ) :
    Shadow σ κ → List (Proposal σ) → Shadow σ κ
  | st, [] => st
  | st, d :: ds => run rq E L sec (step rq E L sec st d).1 ds

/-- "The portfolio is within its caps." -/
structure Inv (E : Rat) (L : Limits) (st : Shadow σ κ) : Prop where
  positions   : st.held.length ≤ L.maxPositions
  gross_le    : st.gross ≤ L.grossCap * E
  gross_nn    : 0 ≤ st.gross
  turn_le     : st.turn ≤ L.turnCap * E
  expo_le     : ∀ s, st.expo s ≤ L.posCap * E
  expo_nn     : ∀ s, 0 ≤ st.expo s
  sec_le      : ∀ k, st.secExpo k ≤ L.sectorCap * E
  sec_nn      : ∀ k, 0 ≤ st.secExpo k

/-! ## Each cap can only shrink the size, and the result respects the cap -/

theorem trim_spec {rq : Rat → Rat} (hrq : RoundDown rq) {cap base p q : Rat}
    (hp : 0 < p) (hq : 0 ≤ q) :
    0 ≤ trim rq cap base p q ∧ trim rq cap base p q ≤ q ∧
    (0 < trim rq cap base p q → base + trim rq cap base p q * p ≤ cap) := by
  unfold trim
  split
  · rename_i hlt
    refine ⟨hrq.nonneg _, ?_, ?_⟩
    · by_cases h0 : 0 < rq ((cap - base) / p)
      · have h1 := hrq.le_of_pos _ h0
        have h2 : (cap - base) / p < q := div_lt_of_lt_mul hp (by grind)
        grind
      · grind
    · intro h0
      have := mul_le_of_le_div hp (hrq.le_of_pos _ h0)
      grind
  · rename_i hge
    refine ⟨hq, Rat.le_refl, fun _ => by grind⟩

theorem cap_mono {base cap p q q' : Rat} (hp : 0 < p) (hqq : q ≤ q')
    (h : base + q' * p ≤ cap) : base + q * p ≤ cap := by
  have := Rat.mul_le_mul_of_nonneg_right hqq (Rat.le_of_lt hp)
  grind

omit [DecidableEq σ] [DecidableEq κ] in
/-- An approved BUY's size satisfies all three caps at once. -/
theorem buySize_caps {rq : Rat → Rat} (hrq : RoundDown rq) {E : Rat} {L : Limits}
    {sec : σ → κ} {st : Shadow σ κ} {d : Proposal σ} (hp : 0 < d.close)
    (hpos : 0 < buySize rq E L sec st d) :
    st.expo d.sym + buySize rq E L sec st d * d.close ≤ L.posCap * E ∧
    st.gross + buySize rq E L sec st d * d.close ≤ L.grossCap * E ∧
    st.secExpo (sec d.sym) + buySize rq E L sec st d * d.close ≤ L.sectorCap * E := by
  unfold buySize at *
  generalize hq0 : rq _ = q0 at *
  have h0 : 0 ≤ q0 := hq0 ▸ hrq.nonneg _
  obtain ⟨n1, l1, c1⟩ := trim_spec (cap := L.posCap * E) (base := st.expo d.sym) hrq hp h0
  generalize trim rq (L.posCap * E) (st.expo d.sym) d.close q0 = q1 at *
  obtain ⟨n2, l2, c2⟩ := trim_spec (cap := L.grossCap * E) (base := st.gross) hrq hp n1
  generalize trim rq (L.grossCap * E) st.gross d.close q1 = q2 at *
  obtain ⟨_, l3, c3⟩ := trim_spec (cap := L.sectorCap * E)
    (base := st.secExpo (sec d.sym)) hrq hp n2
  generalize trim rq (L.sectorCap * E) (st.secExpo (sec d.sym)) d.close q2 = q3 at *
  have hq2 : 0 < q2 := by grind
  have hq1 : 0 < q1 := by grind
  exact ⟨cap_mono hp (by grind) (c1 hq1), cap_mono hp l3 (c2 hq2), c3 hpos⟩

omit [DecidableEq σ] [DecidableEq κ] in
/-- (G2) An approved BUY's size times its stop distance is within the budget. -/
theorem buySize_risk {rq : Rat → Rat} (hrq : RoundDown rq) {E : Rat} {L : Limits}
    {sec : σ → κ} {st : Shadow σ κ} {d : Proposal σ} (hp : 0 < d.close)
    (hsd : 0 < L.stopMult * d.atr) (hpos : 0 < buySize rq E L sec st d) :
    buySize rq E L sec st d * (L.stopMult * d.atr) ≤ E * L.riskPct * ((d.conv : Rat) / 5) := by
  unfold buySize at *
  generalize hb : E * L.riskPct * ((d.conv : Rat) / 5) = b at *
  generalize hq0 : rq (b / (L.stopMult * d.atr)) = q0 at *
  have h0 : 0 ≤ q0 := hq0 ▸ hrq.nonneg _
  obtain ⟨n1, l1, -⟩ := trim_spec (cap := L.posCap * E) (base := st.expo d.sym) hrq hp h0
  generalize trim rq (L.posCap * E) (st.expo d.sym) d.close q0 = q1 at *
  obtain ⟨n2, l2, -⟩ := trim_spec (cap := L.grossCap * E) (base := st.gross) hrq hp n1
  generalize trim rq (L.grossCap * E) st.gross d.close q1 = q2 at *
  obtain ⟨-, l3, -⟩ := trim_spec (cap := L.sectorCap * E)
    (base := st.secExpo (sec d.sym)) hrq hp n2
  generalize trim rq (L.sectorCap * E) (st.secExpo (sec d.sym)) d.close q2 = q3 at *
  have hq0pos : 0 < q0 := by grind
  have hle : q0 ≤ b / (L.stopMult * d.atr) := hq0 ▸ hrq.le_of_pos _ (hq0 ▸ hq0pos)
  have := mul_le_of_le_div hsd hle
  have := Rat.mul_le_mul_of_nonneg_right (show q3 ≤ q0 by grind) (Rat.le_of_lt hsd)
  grind

/-! ## One step -/

theorem hold_never_orders (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ)
    (st : Shadow σ κ) (d : Proposal σ) (h : d.act = .hold) :
    step rq E L sec st d = (st, none) := by
  simp [step, h]

/-- (G3) A SELL is approved only for a position held, and closes exactly the
broker quantity — the gate cannot open a short. -/
theorem sell_only_closes (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ)
    (st : Shadow σ κ) (d : Proposal σ) (h : d.act = .sell) {q : Rat}
    (hq : (step rq E L sec st d).2 = some q) :
    d.sym ∈ st.held ∧ q = d.heldQty ∧ 0 < q := by
  simp only [step, h] at hq
  split at hq <;> try simp at hq
  split at hq <;> simp at hq
  grind

/-- (G2) Risk of an approved BUY between the reference price and the stop. -/
theorem buy_risk_le_budget {rq : Rat → Rat} (hrq : RoundDown rq) {E : Rat} {L : Limits}
    {sec : σ → κ} {st : Shadow σ κ} {d : Proposal σ} (h : d.act = .buy)
    (hm : 0 < L.stopMult) {q : Rat} (hq : (step rq E L sec st d).2 = some q) :
    q * (L.stopMult * d.atr) ≤ E * L.riskPct * ((d.conv : Rat) / 5) := by
  simp only [step, h] at hq
  split at hq; · simp at hq
  split at hq; · simp at hq
  split at hq; · simp at hq
  rename_i hv
  split at hq; · simp at hq
  split at hq; · simp at hq
  rename_i hz _
  simp only [Option.some.injEq] at hq
  subst hq
  have hp : 0 < d.close := by grind
  have hsd : 0 < L.stopMult * d.atr := Rat.mul_pos hm (by grind)
  exact buySize_risk hrq hp hsd (by grind)

/-- (G2, stop) If the position is exited at or above its frozen stop, the loss
is within the budget. `stopPrice` rounds up, so this holds for the real stop. -/
theorem stop_loss_le_budget {q p sd budget exit : Rat} (hq : 0 ≤ q)
    (hrisk : q * sd ≤ budget) (hexit : stopPrice p sd ≤ exit) :
    q * (p - exit) ≤ budget := by
  have h1 := stopPrice_ge p sd
  have h2 : p - exit ≤ sd := by grind
  have := Rat.mul_le_mul_of_nonneg_left h2 hq
  grind

/-- (G1) One step preserves the invariant, whatever the proposal. -/
theorem step_preserves {rq : Rat → Rat} (hrq : RoundDown rq) {E : Rat} {L : Limits}
    {sec : σ → κ} {st : Shadow σ κ} (d : Proposal σ) (hinv : Inv E L st) :
    Inv E L (step rq E L sec st d).1 := by
  obtain ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩ := hinv
  unfold step
  split
  · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
  · -- SELL
    split; · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
    split; · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
    rename_i hturn
    have heh := hen d.sym
    simp only [sellState]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact Nat.le_trans (List.length_erase_le) hpos
    · grind
    · grind
    · grind
    · intro s; simp only [upd]; split <;> grind
    · intro s; simp only [upd]; split <;> grind
    · intro k; simp only [upd]; split
      · have := hs k; have := hsn k; grind
      · exact hs k
    · intro k; simp only [upd]; split <;> grind
  · -- BUY
    split; · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
    split; · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
    rename_i hroom
    split; · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
    rename_i hpx
    split; · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
    rename_i hsz
    split; · exact ⟨hpos, hg, hgn, ht, he, hen, hs, hsn⟩
    rename_i hturn
    have hp : 0 < d.close := by grind
    have hq : 0 < buySize rq E L sec st d := by grind
    obtain ⟨c1, c2, c3⟩ := buySize_caps (sec := sec) (st := st) hrq hp hq
    have hn : 0 < buySize rq E L sec st d * d.close := Rat.mul_pos hq hp
    simp only [buyState]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · split
      · exact hpos
      · simp only [List.length_cons]; grind
    · exact c2
    · grind
    · grind
    · intro s; simp only [upd]; split
      · rename_i hs'; subst hs'; exact c1
      · exact he s
    · intro s; simp only [upd]; split
      · have := hen d.sym; grind
      · exact hen s
    · intro k; simp only [upd]; split
      · rename_i hk; subst hk; exact c3
      · exact hs k
    · intro k; simp only [upd]; split
      · have := hsn k; grind
      · exact hsn k

/-! ## The whole cycle -/

/-- (C) Main theorem. For every list of proposals — every possible output of
the LLM agents and every market state — a cycle that starts within the caps
ends within the caps. -/
theorem cycle_safe {rq : Rat → Rat} (hrq : RoundDown rq) {E : Rat} {L : Limits}
    {sec : σ → κ} (ds : List (Proposal σ)) {st : Shadow σ κ} (hinv : Inv E L st) :
    Inv E L (run rq E L sec st ds) := by
  induction ds generalizing st with
  | nil => exact hinv
  | cons d ds ih => exact ih (step_preserves hrq d hinv)

/-- The deployed instance: Python's rounding. -/
theorem velveteen_cycle_safe {E : Rat} {L : Limits} {sec : σ → κ}
    (ds : List (Proposal σ)) {st : Shadow σ κ} (hinv : Inv E L st) :
    Inv E L (run roundQty E L sec st ds) :=
  cycle_safe roundQty_spec ds hinv

/-! ## The gate before the fix is unsafe -/

/-- The pre-Oct-2026 gate: identical, except that the position count and the
gross exposure used by the checks come from the broker snapshot `snap`, taken
once at the start of the cycle (risk.py used `account.positions`). The state
still records what was actually bought, so the real portfolio is observable. -/
def stepOld (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ) (snap : Shadow σ κ)
    (st : Shadow σ κ) (d : Proposal σ) : Shadow σ κ :=
  match d.act with
  | .buy =>
    -- the checks see the snapshot's positions and gross, not the current ones
    let view : Shadow σ κ := { st with held := snap.held, gross := snap.gross }
    if (step rq E L sec view d).2.isSome then
      buyState sec st d (buySize rq E L sec view d * d.close)
    else st
  | _ => (step rq E L sec st d).1

def runOld (rq : Rat → Rat) (E : Rat) (L : Limits) (sec : σ → κ) (snap : Shadow σ κ) :
    Shadow σ κ → List (Proposal σ) → Shadow σ κ
  | st, [] => st
  | st, d :: ds => runOld rq E L sec snap (stepOld rq E L sec snap st d) ds

/-- Default limits from config.yaml, with the gross cap at 80%. -/
def demoLimits : Limits :=
  { posCap := 1/10, maxPositions := 8, grossCap := 4/5, riskPct := 1/100, stopMult := 2,
    turnCap := 1/4, sectorCap := 3/10, minConv := 3, minNotional := 1 }

/-- Seven positions of $9,800 each (68.6% of $100k), each in its own sector. -/
def demoStart : Shadow Nat Nat :=
  { held := [0, 1, 2, 3, 4, 5, 6], expo := fun s => if s < 7 then 9800 else 0,
    gross := 68600, secExpo := fun k => if k < 7 then 9800 else 0, turn := 0 }

def demoBuy (s : Nat) : Proposal Nat :=
  { sym := s, act := .buy, conv := 5, close := 100, atr := 1, heldQty := 0, vetoed := false }

theorem demoStart_inv : Inv 100000 demoLimits demoStart := by
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, ?_, ?_, ?_, ?_⟩ <;>
  · intro s; simp only [demoStart, demoLimits]; split <;> decide +kernel +kernel

/-- Machine-checked counterexample: from a portfolio within its caps, the old
gate approves two BUYs and ends with 9 positions (cap 8) and 88.6% gross
exposure (cap 80%). -/
theorem old_gate_unsafe :
    ∃ (st : Shadow Nat Nat) (ds : List (Proposal Nat)),
      Inv 100000 demoLimits st ∧ ¬ Inv 100000 demoLimits (runOld roundQty 100000 demoLimits id st st ds) := by
  refine ⟨demoStart, [demoBuy 7, demoBuy 8], demoStart_inv, fun h => ?_⟩
  have := h.positions
  revert this
  decide +kernel

/-- …while the fixed gate, on the same input, stays within the caps
(an instance of `cycle_safe`, checked here by evaluation as well). -/
example : (run roundQty 100000 demoLimits id demoStart [demoBuy 7, demoBuy 8]).held.length = 8 := by
  decide +kernel

end Velveteen
