/-
`Model.FlashAttn` — Lean model of the Metal flash-attention divergence.

Two candidate mechanisms are tested:
  (A) operand/storage rounding — QK^T products ingest f16 mantissa loss;
  (B) ORDER-OF-OPERATIONS catastrophic precision loss — the online-softmax
      rescale (`o *= exp(m_old - m_new)` per KV block) repeatedly shrinks/
      regrows the accumulator, so at f32 (23-bit mantissa) the running
      accumulation loses precision relative to a single global-max softmax,
      and the error compounds with the number of KV blocks (i.e. with T).

We compute three schemes, each at a selectable fraction-bit count
(f64 = 52 "exact", f32 = 23, f16 = 10):
 1. naive        : full S = QK^T/sqrt(d), ONE softmax over ALL keys.
 2. flash-online : running max + per-block rescale (kernel's order).
 3. flash-twopass: blockwise (bounded memory) but find the GLOBAL max first,
                   then a single softmax pass — no per-block accumulator
                   rescale.
If (B) is the culprit: flash-online diverges from naive while flash-twopass
matches naive at the SAME precision, and the gap grows with T.

Run (from verify/):  lake env lean Model/FlashAttn.lean
-/

namespace Model.FlashAttn

def fmax (a b : Float) : Float := if a > b then a else b

/-- emulate a fixed fraction-bit width by grid rounding to multiples of
    2^(e - fb) where e = floor(log2 |x|). fb = fraction bits (23=f32,10=f16).
    fb >= 53 is a true no-op (host Float is f64). -/
def roundFB (fb : Nat) (x : Float) : Float :=
  if fb >= 53 then x
  else if x == 0 then 0 else
    let ax := Float.abs x
    let eF := Float.floor (Float.log ax / Float.log 2.0)
    let stepPow := eF - Float.ofNat fb
    let step := Float.pow 2.0 stepPow
    let q := Float.floor (ax / step + 0.5) * step
    if x < 0 then -q else q

def op (fb : Nat) (x : Float) : Float := roundFB fb x

def cosSim (a b : Array Float) : Float := Id.run do
  let mut dot := 0.0
  let mut na := 0.0
  let mut nb := 0.0
  for i in [:a.size] do
    dot := dot + a[i]!*b[i]!
    na  := na  + a[i]!*a[i]!
    nb  := nb  + b[i]!*b[i]!
  dot / ((Float.sqrt (na*nb)) + 1e-30)

def dotRowAux (Q X : Array (Array Float)) (i j : Nat) : Nat → Float → Float
  | 0, acc => acc
  | k+1, acc => dotRowAux Q X i j k (acc + Q[i]![k]! * X[j]![k]!)

def dotRow (Q X : Array (Array Float)) (i j : Nat) : Float :=
  dotRowAux Q X i j ((Q[i]!).size) 0.0

def scoreOf (fb : Nat) (Q K : Array (Array Float)) (i j : Nat) (scale : Float) : Float :=
  op fb (op fb (dotRow Q K i j) * scale)

def mkZeros (n : Nat) : Array Float := Id.run do
  let mut a : Array Float := #[]
  for _ in [:n] do a := a.push 0.0
  a

def naiveAttn (fb : Nat) (Q K V : Array (Array Float)) : Array Float := Id.run do
  let Tq := Q.size; let Tk := K.size
  let dv := (V[0]!).size
  let scale := 1.0 / Float.sqrt (Float.ofNat (Q[0]!).size)
  let mut out : Array Float := #[]
  for i in [:Tq] do
    let mut mx := -1.0e30
    let mut s : Array Float := #[]
    for j in [:Tk] do
      let v := scoreOf fb Q K i j scale
      s := s.push v
      mx := fmax mx v
    let mut sm := 0.0
    for j in [:Tk] do sm := sm + op fb (Float.exp (op fb (s[j]! - mx)))
    for l in [:dv] do
      let mut acc := 0.0
      for j in [:Tk] do
        let p := op fb (Float.exp (op fb (s[j]! - mx)) / sm)
        acc := acc + op fb (p * V[j]![l]!)
      out := out.push acc
  out

def flashOnline (fb : Nat) (B : Nat) (Q K V : Array (Array Float)) : Array Float := Id.run do
  let Tq := Q.size; let Tk := K.size
  let dv := (V[0]!).size
  let scale := 1.0 / Float.sqrt (Float.ofNat (Q[0]!).size)
  let nblk := (Tk + B - 1) / B
  let mut out : Array Float := #[]
  for i in [:Tq] do
    let mut m : Float := -1.0e30
    let mut l : Float := 0.0
    let mut o := mkZeros dv
    for b in [:nblk] do
      let j0 := b*B
      let j1 := Nat.min (j0 + B) Tk
      let mut bmx : Float := -1.0e30
      let mut bs : Array Float := #[]
      for j in [j0:j1] do
        let v := scoreOf fb Q K i j scale
        bs := bs.push v
        bmx := fmax bmx v
      let mnew := fmax m bmx
      let a := op fb (Float.exp (op fb (m - mnew)))
      for l0 in [:dv] do o := o.set! l0 (op fb (o[l0]! * a))
      l := op fb (l * a)
      let mut bls := 0.0
      for idx in [:bs.size] do bls := bls + op fb (Float.exp (op fb (bs[idx]! - mnew)))
      l := op fb (l + bls)
      for idx in [:bs.size] do
        let p := op fb (Float.exp (op fb (bs[idx]! - mnew)))
        for l0 in [:dv] do
          o := o.set! l0 (op fb (o[l0]! + op fb (p * V[(j0 + idx)]![l0]!)))
      m := mnew
    for l0 in [:dv] do out := out.push (op fb (o[l0]! / l))
  out

def flashTwoPass (fb : Nat) (B : Nat) (Q K V : Array (Array Float)) : Array Float := Id.run do
  let Tq := Q.size; let Tk := K.size
  let dv := (V[0]!).size
  let scale := 1.0 / Float.sqrt (Float.ofNat (Q[0]!).size)
  let mut out : Array Float := #[]
  for i in [:Tq] do
    let mut mx : Float := -1.0e30
    for j in [:Tk] do mx := fmax mx (scoreOf fb Q K i j scale)
    let mut l : Float := 0.0
    for j in [:Tk] do l := l + op fb (Float.exp (op fb (scoreOf fb Q K i j scale - mx)))
    for l0 in [:dv] do
      let mut acc := 0.0
      for j in [:Tk] do
        let p := op fb (Float.exp (op fb (scoreOf fb Q K i j scale - mx)) / l)
        acc := acc + op fb (p * V[j]![l0]!)
      out := out.push acc
  out

def lcgNext (st : Nat) : (Nat × Float) :=
  -- 64-bit-ish LCG on Nat (no overflow unless seed huge; fine for our sizes)
  let ns := (st * 6364136223846793005 + 1442695040888963407)
  let lo : Nat := ns % 1000000
  let f  := (Float.ofNat lo / 1000000.0) * 2.0 - 1.0
  (ns, f)

def randMat' (n d : Nat) (seedbase : Nat) : Array (Array Float) := Id.run do
  let mut s := seedbase + 1
  let mut m : Array (Array Float) := #[]
  for _ in [:n] do
    let mut row : Array Float := #[]
    for _ in [:d] do
      let (s1, f1) := lcgNext s
      s := s1
      row := row.push f1
    m := m.push row
  m

def fbLabel : Nat → String
  | 53 => "f64(53)"
  | 23 => "f32(23)"
  | 10 => "f16(10)"
  | n  => "fb" ++ toString n

def run : IO Unit := do
  let d := 16
  let B := 8
  for fb in [53, 23, 10] do
    IO.println s!"--- fraction bits: {fbLabel fb} ---"
    for T in [32, 64, 128] do
      let Q := randMat' T d 11
      let K := randMat' T d 22
      let V := randMat' T d 33
      let na      := naiveAttn fb Q K V
      let online  := flashOnline fb B Q K V
      let online1 := flashOnline fb T Q K V   -- single giant block == no rescale
      let twopass := flashTwoPass fb B Q K V
      let c1 := (cosSim online na).toString
      let c0 := (cosSim online1 na).toString
      let c2 := (cosSim twopass na).toString
      IO.println
        (s!"  T={T}: online(B={B})={c1} | online(B=T)={c0} | " ++
         s!"twopass={c2} [each vs naive]")

#eval Model.FlashAttn.run
